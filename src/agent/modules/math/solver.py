from ....client import client
from ...state import AgentState
from langchain_experimental.utilities import PythonREPL
import re

# Initialize Python REPL
python_repl = PythonREPL()

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r'```(?:python)?\s*([\s\S]*?)```'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()

def math_solver_node(state: AgentState) -> AgentState:
    """
    Math solver node:
    1. generate_math_code() - vnpt_large: Phân tích đề và sinh code
    2. Execute code using PythonREPL
    3. select_math_answer() - vnpt_small: Tổng hợp kết quả và chọn đáp án
    
    Output: Chỉ trả về một chữ cái (A, B, C, D,...)
    """
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    # Step 1: Generate code using vnpt_large
    code_gen_prompt = f"""Bạn là chuyên gia giải toán bằng lập trình Python.

**Câu hỏi:** {question}

**Các đáp án:**
{choices_str}

**Yêu cầu:**
1. Đọc hiểu đề bài cẩn thận
2. Phân tích từng bước để giải quyết bài toán
3. Viết code Python ngắn gọn, chính xác
4. In ra kết luận, kết quả cuối cùng của bài toán bằng print(), nhớ kết luận đầy đủ đơn vị đo nếu có.

**Lưu ý:**
- Chỉ sử dụng thư viện chuẩn Python (math, statistics, fractions, decimal)
- Code phải print() ra kết quả để so sánh với các đáp án

Hãy viết code Python:"""
    
    print(f"[Math] Step 1: Generating code using generate_math_code()...")
    code_response = client.generate_math_code(code_gen_prompt)
    
    code = extract_code_block(code_response)
    if not code:
        code = code_response
    
    print(f"[Math] Generated code:\n{code[:200]}...")
    
    # Step 2: Execute using PythonREPL
    print(f"[Math] Step 2: Executing code using PythonREPL...")
    try:
        execution_result = python_repl.run(code)
        execution_result = execution_result.strip() if execution_result else "No output"
    except Exception as e:
        execution_result = f"Error: {type(e).__name__}: {str(e)}"
    
    print(f"[Math] Execution result: {execution_result}")
    
    # Step 3: Select answer using vnpt_small
    reasoning_prompt = f"""Bạn là trợ lý chọn đáp án thông minh. Dựa vào kết quả đã tính toán từ bước trước, hãy chọn đáp án đúng.

**Câu hỏi gốc:** {question}

**Các đáp án:**
{choices_str}

**Kết quả tính toán từ code:** {execution_result}

**Yêu cầu:**
- So sánh kết quả với các đáp án
- Suy luận và chọn đáp án có giá trị gần nhất hoặc khớp với kết quả (lưu ý có thể phải đồi đơn vị đo nếu có)
- CHỈ TRẢ LỜI MỘT CHỮ CÁI ĐẠI DIỆN CHO KẾT QUẢ (A, B, C, D, E,...), KHÔNG GIẢI THÍCH"""

    print(f"[Math] Step 3: Selecting answer using select_math_answer()...")
    answer_response = client.select_math_answer(reasoning_prompt)
    
    # Extract answer letter
    answer = answer_response.strip()
    # Tìm đáp án đứng độc lập 1 mình
    match = re.search(r'\b([A-Z])\b', answer, re.IGNORECASE)
    if match:
        # Lấy đáp án độc lập tìm thấy
        state["answer"] = match.group(1).upper()
    else:
        # Nếu không có đáp án độc lập, thử tìm chữ cái đầu tiên (hành vi ban đầu)
        match_any_char = re.search(r'([A-Z])', answer, re.IGNORECASE)
        if match_any_char:
            state["answer"] = match_any_char.group(1).upper()
        else:
            # Nếu không tìm thấy bất kỳ chữ cái nào, mặc định là "A"
            state["answer"] = "A"
    
    state["reasoning"] = f"Code executed. Result: {execution_result}"
    print(f"[Math] Final answer: {state['answer']}")
    
    return state