from langgraph.graph import StateGraph, END
from models.state import ContractState
from nodes.preprocessing import preprocess_node
from nodes.ocr_extraction import ocr_extraction_node
from nodes.indexing import indexing_node
from nodes.extraction_agent import extraction_agent_node
from nodes.excel_generation import excel_generation_node

def build_graph():
    g = StateGraph(ContractState)
    g.add_node("preprocess", preprocess_node)
    g.add_node("ocr_extract", ocr_extraction_node)
    g.add_node("index", indexing_node)
    g.add_node("extract", extraction_agent_node)
    g.add_node("generate_excel", excel_generation_node)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "ocr_extract")
    g.add_edge("ocr_extract", "index")
    g.add_edge("index", "extract")
    g.add_edge("extract", "generate_excel")
    g.add_edge("generate_excel", END)
    return g.compile()

# Singleton – imported by app.py
contract_graph = build_graph()

def save_graph_visualization(path: str = "graph_workflow.png") -> None:
    png_bytes = contract_graph.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png_bytes)
    print(f"Graph saved to {path}")

if __name__ == "__main__":
    save_graph_visualization()