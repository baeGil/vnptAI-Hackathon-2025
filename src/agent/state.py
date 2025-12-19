from typing import TypedDict, Annotated, List, Optional
import operator


class AgentState(TypedDict):
    """
    LangGraph agent state for VNPT AI Hackathon pipeline.
    
    Follows LangGraph's standard TypedDict pattern for state management.
    All nodes receive and return this state structure.
    """
    # Input fields
    question: str           # The question text
    qid: str               # Question ID
    choices: List[str]     # List of answer choices (e.g., ["A. option", "B. option"])
    
    # Router output
    category: str          # Question category: 'math', 'rag', 'reading', 'toxic'
    
    # Context and retrieval
    context: str           # Retrieved context for RAG/Reading
    
    # Output fields
    answer: str            # Final answer: 'A', 'B', 'C', 'D', etc.
    reasoning: str         # Reasoning explanation

