from ..client import client
from .state import AgentState

def router_node(state: AgentState) -> AgentState:
    """
    Router node using vnpt_hackathon_small for classification.
    """
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    classification_prompt = f"""Bạn là một trợ lý phân loại câu hỏi. Nhiệm vụ của bạn là hãy phân loại câu hỏi sau vào MỘT trong các nhóm:

- TOXIC: Khi các đáp án xuất hiện ít nhất 1 đáp án kiểu từ chối trả lời hoặc nằm ngoài phạm vi trả lời như: "tôi không thể trả lời", "tôi không biết", "tôi không thể cung cấp", "không thể chia sẻ", "không thể hỗ trợ", "nằm ngoài phạm vi trả lời", "nằm ngoài tầm hiểu biết", etc.
- MATH: Câu hỏi không phải là câu hỏi lý thuyết về định nghĩa, sử dụng kiến thức mà là câu hỏi yêu cầu thực hành và tư duy logic, giải toán, giải lý, tính toán, bài toán đếm, tổ hợp, đạo hàm, vi phân, thống kê, etc. hoặc có thể giải quyết bằng lập trình/code, giải thuật, etc.
- READING: Câu hỏi được cung cấp sẵn một đoạn văn/đoạn thông tin (ví dụ: "Đoạn thông tin:", "Đoạn văn:", "Cho đoạn văn sau:") và yêu cầu đọc hiểu để trả lời.
- RAG: Câu hỏi cần tra cứu thêm kiến thức bên ngoài, kiến thức tổng hợp, lịch sử, địa lý, văn hóa, chính trị, hoặc không thuộc 3 loại trên.

Câu hỏi:
{question}

Các đáp án:
{choices_str}

Chỉ trả về MỘT từ duy nhất: TOXIC, MATH, READING, hoặc RAG tương ứng với nhóm câu hỏi mà bạn đã phân loại."""

    print(f"[Router] Classifying using classify_router()...")
    response = client.classify_router(classification_prompt)
    
    response_upper = response.strip().upper()
    
    if "TOXIC" in response_upper:
        state["category"] = "toxic"
    elif "MATH" in response_upper:
        state["category"] = "math"
    elif "READING" in response_upper:
        state["category"] = "reading"
    else:
        state["category"] = "rag"
    
    print(f"[Router] -> {state['category'].upper()} (response: '{response.strip()}')")
    return state
