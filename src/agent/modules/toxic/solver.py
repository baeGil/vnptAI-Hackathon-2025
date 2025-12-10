from ...state import AgentState
from ....logger import log
import re

TOXIC_KEYWORDS = [
    "tôi không thể trả lời",
    "tôi không biết",
    "tôi không thể cung cấp",
    "không thể chia sẻ",
    "không thể hỗ trợ",
    "không được phép",
    "từ chối trả lời",
    "không thể đưa ra",
    "không cung cấp",
]

def toxic_checker_node(state: AgentState) -> AgentState:
    """Toxic checker: find refusal answer directly."""
    choices = state["choices"]
    
    log(f"[Toxic] Checking for refusal pattern...")
    
    for choice in choices:
        choice_lower = choice.lower()
        for keyword in TOXIC_KEYWORDS:
            if keyword in choice_lower:
                match = re.match(r'^([A-J])\.\s*', choice)
                if match:
                    state["answer"] = match.group(1)
                    state["reasoning"] = f"Toxic: {choice[:50]}..."
                    log(f"[Toxic] Answer: {state['answer']}")
                    return state
    
    # Fallback
    for choice in choices:
        match = re.match(r'^([A-J])\.\s*', choice)
        if match:
            state["answer"] = match.group(1)
            break
    else:
        state["answer"] = "A"
    
    state["reasoning"] = "Toxic category but no clear refusal found."
    log(f"[Toxic] Fallback answer: {state['answer']}")
    return state