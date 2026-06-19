import csv
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

from models.state import ContractState
from services.llm import LLMService
from services.vector_store import HybridVectorStore
from config.settings import EXTRACT_FIELDS, TOP_K_RETRIEVAL, OUTPUT_DIR, FIELD_DISPLAY_NAMES



FIELD_QUERIES: Dict[str, List[str]] = {
    "party_a_legal_name": [
        "party initiating issuing commissioning contract name",
        "contracting authority entity procuring services",
        "issued by letterhead header organization name",
        "client buyer purchaser government agency department",
    ],
    "party_b_legal_name": [
        "party providing fulfilling goods services name",
        "vendor contractor manufacturer service provider legal name",
        "signed by fulfilling party company registration name",
        "seller supplier respondent awarded contractor",
    ],
    "start_date": [
        "contract start date commencement effective date",
        "agreement begins from date contract period start",
        "effective from date of contract",
    ],
    "end_date": [
        "contract end date expiry termination date",
        "agreement expires valid until contract period end",
        "contract completion date",
    ],
    "price_details": [
        "rate card price list pricing table fees charges",
        "unit price amount cost per item service rate",
        "pricing schedule tariff commercial rates",
        "fee structure price per unit volume pricing",
    ],
    "payment_timeline": [
        "payment terms net days due net 30 net 60 payment schedule",
        "invoice settlement period days due payment deadline",
        "payment due within days of invoice",
    ],
    "payment_conditions": [
        "payment conditions triggers acceptance criteria invoice rules",
        "payment upon delivery acceptance milestone conditions",
        "invoice approval payment release conditions terms",
    ],
}



FIELD_PROMPTS: Dict[str, str] = {
    "party_a_legal_name": (
        "Extract the full legal name of the party INITIATING or COMMISSIONING this contract — "
        "the entity that is buying, procuring, or engaging the other party. "
        "This party is typically the one who originated or issued the contract. "
        "They may appear in the document header, letterhead, 'issued by' block, or signature block "
        "rather than being explicitly labeled. In government contracts this is often a state agency "
        "or department. In commercial contracts this is typically the client or buyer. "
        "Return the most formally stated version of their full legal name."
    ),
    "party_b_legal_name": (
        "Extract the full legal name of the party FULFILLING or PROVIDING under this contract — "
        "the entity that is supplying goods, delivering services, or performing the work. "
        "This party is typically the one responding to or awarded the contract. "
        "Look in the manufacturer or vendor information block, signature block, "
        "or anywhere a company name appears alongside an obligation to deliver. "
        "Return the most formally stated version of their full legal name."
    ),
    "start_date": (
        "Extract the contract start / commencement / effective date. "
        "Look for 'Start Date:', 'Effective Date:', 'Commencement Date:', 'From:', "
        "'Contract Period begins:'. Normalise to YYYY-MM-DD if possible."
    ),
    "end_date": (
        "Extract the contract end / expiry / termination date. "
        "Look for 'End Date:', 'Expiry Date:', 'Termination Date:', 'Valid Until:', "
        "'Contract Period ends:'. Normalise to YYYY-MM-DD if possible."
    ),
    "price_details": (
        "Extract the complete rate card / pricing table from this contract page. "
        "This may include service items, product SKUs, unit prices, currencies, volume tiers, "
        "or any other commercial pricing information. The schema is dynamic – capture every "
        "column and row as found in the document."
    ),
    "payment_timeline": (
        "Extract only the payment timeline — the specific net payment period or day count. "
        "Look for values like 'Net 30', 'Net 60', '45 days', '30 days from invoice date'. "
        "Return the shortest unambiguous value (e.g. 'Net 30', '60 days'). "
        "Return null if no explicit day count or net term is stated."
    ),
    "payment_conditions": (
        "Extract the full narrative of payment conditions — everything beyond the simple day count. "
        "This includes: triggers for payment (e.g. delivery, acceptance, milestone completion), "
        "invoice submission rules, acceptance criteria, approval workflows, early payment discounts, "
        "and late payment penalties. Capture the complete clause text relevant to when and how "
        "payment is released."
    ),
}

# Fields that must always anchor page 0 (cover/header page) 
# Party identity fields often live in the letterhead or first-page header,
# which can be filtered out by score threshold if interior pages score higher.

PAGE_0_ANCHOR_FIELDS = {"party_a_legal_name", "party_b_legal_name"}

# ── Retrieval stats dataclass (plain dict for simplicity)

def _empty_stats(field: str) -> Dict[str, Any]:
    return {"field": field, "chunks_retrieved": 0, "pages_passed": 0, "page_numbers": []}


# Retrieval 

FIELD_K_OVERRIDES: Dict[str, int] = {
    "price_details": 20,  # rate cards often span multiple pages
}


def _retrieve_top_chunks(field: str, store: HybridVectorStore) -> List[Dict]:
    """Multi-query retrieval with deduplication, returns top-k child chunks.

    For party identity fields, page 0 (cover/letterhead) is always anchored
    regardless of its score, so the issuing authority is never filtered out.
    """
    k_cap = FIELD_K_OVERRIDES.get(field, TOP_K_RETRIEVAL)
    k_per_query = max(6, k_cap // len(FIELD_QUERIES[field]))

    seen: set = set()
    results: List[Dict] = []
    for q in FIELD_QUERIES[field]:
        for r in store.hybrid_search(q, k=k_per_query):
            key = r["chunk"][:80]
            if key not in seen:
                seen.add(key)
                results.append(r)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Anchor page 0 for party fields — cover page / letterhead must always
    # be included regardless of score, as party names often live only there.
    if field in PAGE_0_ANCHOR_FIELDS:
        page_0_chunks = store.hybrid_search("contract issued by party name letterhead", k=4)
        for r in page_0_chunks:
            if r["metadata"].get("page", 999) == 0:
                key = r["chunk"][:80]
                if key not in seen:
                    seen.add(key)
                    results.append(r)

    return results[:k_cap]


def _build_parent_context(chunks: List[Dict]) -> Tuple[str, List[int]]:
    """Expand child chunks to their full parent pages (deduplicated, score-filtered).

    Only pages whose best chunk score is >= 50% of the top score are included.
    Returns the context string and the sorted list of unique page numbers (1-indexed).
    """
    page_best_score: Dict[int, float] = {}
    page_texts: Dict[int, str] = {}

    for r in chunks:
        pg = r["metadata"].get("page", 0)
        score = r.get("score", 0.0)
        if score > page_best_score.get(pg, -1.0):
            page_best_score[pg] = score
        if pg not in page_texts:
            page_texts[pg] = r["metadata"].get("page_text") or r["chunk"]

    if page_best_score:
        top_score = max(page_best_score.values())
        threshold = top_score * 0.5
        relevant_pages = {pg for pg, sc in page_best_score.items() if sc >= threshold}
    else:
        relevant_pages = set(page_texts)

    filtered = {pg: page_texts[pg] for pg in relevant_pages}
    context = "\n\n".join(
        f"[Full page {pg + 1}]\n{text}"
        for pg, text in sorted(filtered.items())
    )
    page_numbers = sorted(pg + 1 for pg in filtered)
    return context, page_numbers


def _candidate_pages_for_price(chunks: List[Dict]) -> Dict[int, str]:
    """Return pages to use for per-page price extraction.

    Keeps pages whose best chunk score is >= 30% of the top score — a loose
    filter that removes true noise while preserving all pricing pages.
    """
    page_best_score: Dict[int, float] = {}
    page_texts: Dict[int, str] = {}

    for r in chunks:
        pg = r["metadata"].get("page", 0)
        score = r.get("score", 0.0)
        if score > page_best_score.get(pg, -1.0):
            page_best_score[pg] = score
        if pg not in page_texts:
            page_texts[pg] = r["metadata"].get("page_text") or r["chunk"]

    if not page_best_score:
        return page_texts

    top_score = max(page_best_score.values())
    threshold = top_score * 0.3
    return {
        pg: page_texts[pg]
        for pg, sc in page_best_score.items()
        if sc >= threshold
    }


# Extraction 
def _canonicalize_columns(
    page_schemas: List[Dict[str, Any]],
    llm: LLMService,
) -> Tuple[Dict[str, str], str, str, int, int]:
    """One LLM call to unify column name variants across pages.

    Sends columns + one sample row per page so the LLM can compare actual
    values (e.g. 'SKU: ECSK02010' vs 'Item Number: ECSK02020') to decide
    whether columns are the same concept.

    Returns (variant→canonical mapping, input_tokens, output_tokens).
    """
    import json
    prompt = f"""These column headers were extracted from different pages of the same pricing document.
Some columns are the same concept named differently across pages (e.g. "Item Numbers", "Item Number", "SKU").

Page schemas with one sample row each:
{json.dumps(page_schemas, indent=2)}

Return a flat JSON object mapping every column name variant to its canonical name.
- Group columns that represent the same concept into one canonical name
- Use the most descriptive and clearest name as the canonical
- If a column is already unique and clear, map it to itself
- Map every column that appears in the input

Return JSON: {{"<variant>": "<canonical>", ...}}"""

    response, raw_output, in_tok, out_tok = llm.generate_json_tracked(prompt)
    if not response or not isinstance(response, dict):
        all_cols = [c for s in page_schemas for c in s.get("columns", [])]
        return {c: c for c in all_cols}, prompt, raw_output, in_tok, out_tok
    return response, prompt, raw_output, in_tok, out_tok


def _extract_price_details(
    chunks: List[Dict],
    llm: LLMService,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Extract price details page-by-page then merge all rows.

    Returns (extraction result, retrieval stats, prompt_log_entry).
    """
    candidate_pages = _candidate_pages_for_price(chunks)

    all_rows: List[Any] = []
    page_schemas: List[Dict[str, Any]] = []
    contributing_pages: List[int] = []
    first_raw_text: str = ""
    page_calls: List[Dict[str, Any]] = []
    total_in_tok = 0
    total_out_tok = 0

    for pg, page_text in sorted(candidate_pages.items()):
        prompt = f"""{FIELD_PROMPTS["price_details"]}

[Page {pg + 1}]
{page_text}

Return JSON:
{{
  "columns": ["<exact col header 1>", "<exact col header 2>", ...],
  "value": [
    {{"<exact col header 1>": "...", "<exact col header 2>": "..."}}
  ],
  "confidence": <0.0–1.0>,
  "raw_text": "<short verbatim snippet>"
}}

Rules:
- "columns": list the exact column headers as they appear in the document table
- "value": use ONLY the names from "columns" as keys — do not invent new key names
- Extract EVERY pricing row present on this page
- If this page has no pricing data: {{"columns": [], "value": [], "confidence": 0.0, "raw_text": ""}}"""

        response, raw_output, in_tok, out_tok = llm.generate_json_tracked(prompt)
        total_in_tok += in_tok
        total_out_tok += out_tok
        page_calls.append({
            "page": pg + 1,
            "prompt": prompt,
            "prompt_output": raw_output,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })

        if response and response.get("value"):
            raw_cols = response.get("columns", [])
            col_map = {c: c.strip().title() for c in raw_cols}
            normalized_page_rows = [
                {col_map.get(k, k.strip().title()): v for k, v in row.items()}
                for row in response["value"]
            ]
            page_schemas.append({
                "page": pg + 1,
                "columns": list(col_map.values()),
                "sample": normalized_page_rows[0] if normalized_page_rows else {},
            })
            all_rows.extend(normalized_page_rows)
            contributing_pages.append(pg + 1)
            if not first_raw_text:
                first_raw_text = response.get("raw_text", "")

    stats = {
        "field": "price_details",
        "chunks_retrieved": len(chunks),
        "pages_passed": len(candidate_pages),
        "page_numbers": sorted(pg + 1 for pg in candidate_pages),
    }

    canon_call: Dict[str, Any] = {}

    if not all_rows:
        return {"columns": [], "value": [], "confidence": 0.0, "page_ref": None, "raw_text": ""}, stats, {
            "attribute": "price_details",
            "display_name": FIELD_DISPLAY_NAMES["price_details"],
            "function": "_extract_price_details",
            "rag_queries": FIELD_QUERIES["price_details"],
            "top_k_chunks": [],
            "page_calls": page_calls,
            "canon_call": canon_call,
            "input_tokens": total_in_tok,
            "output_tokens": total_out_tok,
        }

    # Canonicalize column names across pages when schemas differ
    col_sets = [frozenset(s["columns"]) for s in page_schemas]
    if len(page_schemas) > 1 and len(set(col_sets)) > 1:
        canon_map, c_prompt, c_raw, c_in, c_out = _canonicalize_columns(page_schemas, llm)
        total_in_tok += c_in
        total_out_tok += c_out
        canon_call = {
            "mapping": canon_map,
            "prompt": c_prompt,
            "prompt_output": c_raw,
            "input_tokens": c_in,
            "output_tokens": c_out,
        }
        all_rows = [{canon_map.get(k, k): v for k, v in row.items()} for row in all_rows]

    # Rebuild ordered unique columns from the (possibly remapped) rows
    seen_final: set = set()
    all_columns: List[str] = []
    for row in all_rows:
        for col in row:
            if col not in seen_final:
                seen_final.add(col)
                all_columns.append(col)

    prompt_log_entry = {
        "attribute": "price_details",
        "display_name": FIELD_DISPLAY_NAMES["price_details"],
        "function": "_extract_price_details",
        "rag_queries": FIELD_QUERIES["price_details"],
        "top_k_chunks": [
            {
                "text": c["chunk"][:300],
                "score": round(c.get("score", 0.0), 4),
                "page": c["metadata"].get("page", 0) + 1,
                "faiss_rank": c.get("faiss_rank"),
                "bm25_rank": c.get("bm25_rank"),
            }
            for c in chunks
        ],
        "page_calls": page_calls,
        "canon_call": canon_call,
        "input_tokens": total_in_tok,
        "output_tokens": total_out_tok,
    }

    normalized_rows = [{col: row.get(col, "") for col in all_columns} for row in all_rows]
    page_ref = contributing_pages[0] if len(contributing_pages) == 1 else contributing_pages

    return {
        "columns": all_columns,
        "value": normalized_rows,
        "confidence": 1.0,
        "page_ref": page_ref,
        "raw_text": first_raw_text,
    }, stats, prompt_log_entry


def _extract_field(
    field: str,
    store: HybridVectorStore,
    llm: LLMService,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Returns (extraction result, retrieval stats, prompt_log_entry)."""
    top_chunks = _retrieve_top_chunks(field, store)

    if field == "price_details":
        return _extract_price_details(top_chunks, llm)

    context, page_numbers = _build_parent_context(top_chunks)

    stats = {
        "field": field,
        "chunks_retrieved": len(top_chunks),
        "pages_passed": len(page_numbers),
        "page_numbers": page_numbers,
    }

    prompt = f"""{FIELD_PROMPTS[field]}

Contract pages:
{context}

Return JSON:
{{
  "value": "<extracted value or null>",
  "confidence": <0.0–1.0>,
  "page_ref": <page number or null>,
  "raw_text": "<short verbatim snippet>"
}}

If not found: {{"value": null, "confidence": 0.0, "page_ref": null, "raw_text": ""}}"""

    response, raw_output, in_tok, out_tok = llm.generate_json_tracked(prompt)

    if not response or "value" not in response:
        response = {"value": None, "confidence": 0.0, "page_ref": None, "raw_text": ""}
    elif response.get("page_ref") is None and top_chunks:
        response["page_ref"] = top_chunks[0]["metadata"].get("page", 0) + 1

    prompt_log_entry = {
        "attribute": field,
        "display_name": FIELD_DISPLAY_NAMES.get(field, field),
        "function": "_extract_field",
        "rag_queries": FIELD_QUERIES.get(field, []),
        "top_k_chunks": [
            {
                "text": c["chunk"][:300],
                "score": round(c.get("score", 0.0), 4),
                "page": c["metadata"].get("page", 0) + 1,
                "faiss_rank": c.get("faiss_rank"),
                "bm25_rank": c.get("bm25_rank"),
            }
            for c in top_chunks
        ],
        "prompt": prompt,
        "prompt_output": raw_output,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    return response, stats, prompt_log_entry


#  Retrieval log helpers 

_CSV_HEADER = ["timestamp", "document", "field", "display_name",
               "chunks_retrieved", "pages_passed", "page_numbers"]


def _print_retrieval_table(all_stats: List[Dict[str, Any]], document: str) -> None:
    col = "{:<28} {:>16} {:>12} {}"
    header = col.format("Field", "Chunks Retrieved", "Pages Passed", "Page Numbers")
    divider = "-" * len(header)
    print(f"\n{'='*60}")
    print(f"  Retrieval log — {document}")
    print(f"{'='*60}")
    print(header)
    print(divider)
    for s in all_stats:
        pages_str = str(s["page_numbers"]) if s["page_numbers"] else "[]"
        print(col.format(
            FIELD_DISPLAY_NAMES.get(s["field"], s["field"]),
            s["chunks_retrieved"],
            s["pages_passed"],
            pages_str,
        ))
    print(f"{'='*60}\n")


def _write_csv(all_stats: List[Dict[str, Any]], document: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(OUTPUT_DIR) / f"retrieval_log_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        for s in all_stats:
            writer.writerow({
                "timestamp": ts,
                "document": document,
                "field": s["field"],
                "display_name": FIELD_DISPLAY_NAMES.get(s["field"], s["field"]),
                "chunks_retrieved": s["chunks_retrieved"],
                "pages_passed": s["pages_passed"],
                "page_numbers": ";".join(str(p) for p in s["page_numbers"]),
            })
    return csv_path


#  Node 

def extraction_agent_node(state: ContractState) -> dict:
    from nodes.indexing import get_session_store  # avoids circular import at module level

    log = list(state.get("processing_log", []))
    store = get_session_store()
    llm = LLMService()
    extracted: Dict[str, Any] = {}
    all_stats: List[Dict[str, Any]] = []
    prompt_log: List[Dict[str, Any]] = []
    document = state.get("original_filename", "contract")

    for field in EXTRACT_FIELDS:
        try:
            result, stats, log_entry = _extract_field(field, store, llm)
            extracted[field] = result
            all_stats.append(stats)
            prompt_log.append(log_entry)
        except Exception as e:
            extracted[field] = {
                "value": None,
                "confidence": 0.0,
                "page_ref": None,
                "raw_text": "",
                "error": str(e),
            }
            all_stats.append(_empty_stats(field))
            prompt_log.append({
                "attribute": field,
                "display_name": FIELD_DISPLAY_NAMES.get(field, field),
                "function": "_extract_field",
                "error": str(e),
                "rag_queries": [],
                "top_k_chunks": [],
                "prompt": "",
                "prompt_output": "",
                "input_tokens": 0,
                "output_tokens": 0,
            })

    _print_retrieval_table(all_stats, document)
    csv_path = _write_csv(all_stats, document)

    found = sum(1 for v in extracted.values() if v.get("value"))
    log.append(f"Extraction complete: {found}/{len(EXTRACT_FIELDS)} fields found")
    log.append(f"Retrieval log saved: {csv_path}")

    return {
        **state,
        "extracted_fields": extracted,
        "prompt_log": prompt_log,
        "processing_log": log,
        "current_step": "validation",
    }