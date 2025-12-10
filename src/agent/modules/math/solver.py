from ....client import client
from ...state import AgentState
from ....logger import log
from langchain_experimental.utilities import PythonREPL
import re

python_repl = PythonREPL()

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r'```(?:python)?\s*([\s\S]*?)```'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()

def math_solver_node(state: AgentState) -> AgentState:
    """Math solver: large model generates code, execute, small model selects answer."""
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    code_gen_prompt = f"""Hãy viết code Python để giải quyết bài toán sau.
Yêu cầu:
- Gán kết quả cuối cùng vào biến `result`
- Chỉ sử dụng thư viện chuẩn Python (math, statistics, fractions, decimal)
- Code ngắn gọn và chính xác

Câu hỏi: {question}

```python
"""
    
    log(f"[Math] Step 1: Generating code...")
    code_response = client.generate_math_code(code_gen_prompt)
    
    code = extract_code_block(code_response)
    if not code:
        code = code_response
    
    log(f"[Math] Generated code: {code[:150]}...")
    
    log(f"[Math] Step 2: Executing code...")
    try:
        execution_result = python_repl.run(code)
        execution_result = execution_result.strip() if execution_result else "No output"
    except Exception as e:
        execution_result = f"Error: {type(e).__name__}: {str(e)}"
    
    log(f"[Math] Execution result: {execution_result[:100]}")
    
    reasoning_prompt = f"""Dựa vào kết quả tính toán, hãy chọn đáp án chính xác nhất.

Câu hỏi: {question}

Các đáp án:
{choices_str}

Kết quả tính toán: {execution_result}

CHỈ TRẢ LỜI MỘT CHỮ CÁI (A, B, C, D,...) KHÔNG GIẢI THÍCH"""

    log(f"[Math] Step 3: Selecting answer...")
    answer_response = client.select_math_answer(reasoning_prompt)
    
    answer = answer_response.strip().upper()
    for char in answer:
        if char in 'ABCDEFGHIJ':
            state["answer"] = char
            break
    else:
        state["answer"] = "A"
    
    state["reasoning"] = f"Code executed. Result: {execution_result[:100]}"
    log(f"[Math] Final answer: {state['answer']}")
    
    return state
