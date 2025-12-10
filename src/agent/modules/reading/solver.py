from ....client import client
from ...state import AgentState
import re

def reading_solver_node(state: AgentState) -> AgentState:
    """
    Reading comprehension solver.
    Uses: client.generate_reading_answer() - vnpt_large
    
    Input: Câu hỏi đã có đoạn văn + đáp án đã format (A. xxx, B. xxx,...)
    Output: Chỉ trả về một chữ cái (A, B, C, D,...)
    """
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    prompt = f"""Bạn là chuyên gia đọc hiểu văn bản.

**Nhiệm vụ:** Đọc kỹ đoạn văn trong câu hỏi, phân tích nội dung, và chọn đáp án chính xác nhất.

**Câu hỏi và đoạn văn:**
{question}

**Các đáp án:**
{choices_str}

**Yêu cầu:**
1. Đọc kỹ đoạn văn được cung cấp trong câu hỏi
2. Xác định thông tin cần thiết để trả lời
3. So sánh với từng đáp án
4. Chọn đáp án chính xác nhất

**CHỈ TRẢ LỜI MỘT CHỮ CÁI (A, B, C, D, E,...) KHÔNG GIẢI THÍCH**"""
    
    print(f"[Reading] Processing using generate_reading_answer()...")
    response = client.generate_reading_answer(prompt)
    
    # Extract answer letter
    answer = response.strip().upper()
    for char in answer:
        if char in 'ABCDEFGHIJ':
            state["answer"] = char
            break
    else:
        state["answer"] = "A"
    
    state["reasoning"] = f"Reading comprehension: {response[:100]}"
    print(f"[Reading] Answer: {state['answer']}")
    
    return state
