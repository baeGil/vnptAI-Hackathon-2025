from ...state import AgentState
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
    """
    Toxic checker node:
    - Since toxic is classified by LLM detecting refusal patterns in answers,
      we find and return the refusal answer directly.
    """
    choices = state["choices"]
    
    # Find the answer that contains the refusal
    for choice in choices:
        choice_lower = choice.lower()
        for keyword in TOXIC_KEYWORDS:
            if keyword in choice_lower:
                # Extract the letter from the choice (e.g., "B. Tôi không thể..." -> "B")
                match = re.match(r'^([A-J])\.\s*', choice)
                if match:
                    state["answer"] = match.group(1)
                    state["reasoning"] = f"Toxic question detected. Answer is the refusal: {choice[:50]}..."
                    print(f"[Toxic] Answer: {state['answer']} ({choice[:30]}...)")
                    return state
    
    # Fallback if somehow no refusal found
    # Try to find first choice that looks like a refusal
    for choice in choices:
        match = re.match(r'^([A-J])\.\s*', choice)
        if match:
            state["answer"] = match.group(1)
            break
    else:
        state["answer"] = "A"
    
    state["reasoning"] = "Toxic category but no clear refusal found."
    print(f"[Toxic] Fallback answer: {state['answer']}")
    return state