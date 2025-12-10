from langgraph.graph import StateGraph, END
from .state import AgentState
from .router import router_node
from .modules.math import math_solver_node
from .modules.rag import rag_solver_node
from .modules.reading import reading_solver_node
from .modules.toxic import toxic_checker_node

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("math_solver", math_solver_node)
    workflow.add_node("rag_solver", rag_solver_node)
    workflow.add_node("reading_solver", reading_solver_node)
    workflow.add_node("toxic_checker", toxic_checker_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Conditional routing based on category
    workflow.add_conditional_edges(
        "router",
        lambda state: state["category"],
        {
            "math": "math_solver",
            "rag": "rag_solver",
            "reading": "reading_solver",
            "toxic": "toxic_checker"
        }
    )
    
    # All solvers terminate at END
    workflow.add_edge("math_solver", END)
    workflow.add_edge("rag_solver", END)
    workflow.add_edge("reading_solver", END)
    workflow.add_edge("toxic_checker", END)
    
    return workflow.compile()

app = build_graph()
