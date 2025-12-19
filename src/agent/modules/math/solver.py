from ....client import client
from ....answer import extract_and_normalize
from ...state import AgentState
from ....logger import log
import sys
from io import StringIO
import contextlib
import traceback
import re

class SafePythonExecutor:
    """
    Executes Python code in a safe, isolated namespace.
    Uses a single dictionary for globals and locals to ensure correct scoping 
    for imports and function definitions (mimicking module-level scope).
    """
    def __init__(self):
        self.globals = {}  # Persistent state if needed, or reset per run
        
    def run(self, code: str) -> str:
        # Create a fresh namespace for each run to avoid pollution
        # Use single dict for both globals and locals to fix scoping issues
        namespace = {}
        
        # Capture stdout
        output = StringIO()
        
        try:
            with contextlib.redirect_stdout(output):
                exec(code, namespace, namespace)
            result = output.getvalue().strip()
            return result
        except Exception:
            # Return the full traceback
            return traceback.format_exc()

# Initialize executor
python_executor = SafePythonExecutor()

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r'```(?:python)?\s*([\s\S]*?)```'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()


def _validate_code_syntax(code: str) -> tuple:
    """Check if code has valid Python syntax. Returns (is_valid, error_message)."""
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def _is_placeholder_code(code: str) -> bool:
    """Check if code contains placeholders or is incomplete."""
    if not code or len(code.strip()) < 10:
        return True
    if "..." in code:
        return True
    # Check for {key}-style placeholders (but not f-string or dict literals)
    if re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", code):
        # Exclude common dict/set patterns and f-strings
        if not re.search(r'["\'][^"\']*\{[a-zA-Z_]', code):
            return True
    return False

def _fallback_text_reasoning(question: str, choices: list) -> str:
    """
    Fallback: Use Chain-of-Thought reasoning when code execution consistently fails.
    Returns answer letter.
    """
    choices_str = "\n".join(choices)
    
    prompt = f"""Bạn là chuyên gia giải toán. Code thực thi đã thất bại nhiều lần.
Hãy dùng suy luận logic và kiến thức toán học để giải bài và chọn đáp án đúng.

Câu hỏi: {question}

Các đáp án:
{choices_str}

Hãy suy luận từng bước:
1. Phân tích đề bài
2. Xác định công thức/phương pháp cần dùng
3. Tính toán/suy luận
4. So sánh với các đáp án
5. Chọn đáp án đúng nhất

Cuối cùng, trả lời MỘT CHỮ CÁI: Đáp án: X"""
    
    log(f"[Math] Using CoT fallback (code execution failed)")
    response = client.select_math_answer(prompt)
    
    # Use robust answer extraction
    answer = extract_and_normalize(response, len(choices), default="A")
    return answer

# Math solver node with code generation and execution method
def math_solver_node(state: AgentState) -> AgentState:
    """Math solver: large model generates code, execute, small model selects answer."""
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    code_gen_prompt = f"""
Bạn là chuyên gia giải toán bằng cách viết code Python chuyên nghiệp có độ chính xác tuyệt đối.
Nhiệm vụ: đọc hiểu đề bài và các đáp án được cung cấp để lựa chọn, phân tích yêu cầu, sau đó sinh ra đoạn code Python hoàn chỉnh để giải bài toán.

ĐỀ BÀI:
{question}

CÁC ĐÁP ÁN:
{choices_str}

QUY TRÌNH SUY LUẬN:
Trước khi sinh code, hãy tự động thực hiện 5 bước nội bộ:
1) Hiểu đề bài và các đáp án được cung cấp để lựa chọn: xác định đầu vào, đầu ra, công thức, các trường hợp biên, các trường hợp đặc biệt.  
2) Chọn phương pháp giải phù hợp.
3) Xây dựng thuật toán rõ ràng.  
4) Viết code Python tối ưu và đúng chuẩn.
5) Trả về kết luận đầy đủ của bài toán kèm theo kết quả thực thi dưới dạng LaTeX, để cung cấp đầy đủ ngữ cảnh bài toán cho người dùng.

YÊU CẦU BẮT BUỘC:

1. Code phải:
   - Bắt buộc phải import đầy đủ thư viện cần dùng (nếu có) để tránh lỗi 
   - Bắt buộc phải định nghĩa đầy đủ các biến cần thiết (nếu có) để tránh lỗi
   - Không lỗi runtime, lỗi cú pháp có thể dự đoán trước.
   - Đúng với yêu cầu của đề bài.

2. Quy tắc bắt buộc:
   - Kiểm tra các điều kiện cần và đủ của lý thuyết (nếu có) để tránh sai lệch (ví dụ: Nếu cần tìm cực trị thì giải f'(x)=0 chưa đủ, phải kiểm tra f''(x); Nếu cần tìm điểm uốn thì giải f''(x)=0 chưa đủ, phải kiểm tra f''(x) có đổi dấu không)
   - Nguyên tắc "Standard Types": Luôn ưu tiên chuyển đổi các object phức tạp của thư viện (SymPy Matrix, Set, Interval...) về các kiểu dữ liệu cơ bản của Python (List, Dict, Int, Float, String) trước khi trả về kết quả.
   - Giá trị riêng (eigenvalue): Nếu tìm giá trị riêng của ma trận, BẮT BUỘC phải liệt kê đầy đủ theo bội số (multiplicity). Ví dụ: nếu eigenvalue=2 có bội số 3 thì phải in ra [2, 2, 2] chứ KHÔNG chỉ trả về [2].
   - Kết quả dạng phân số: Nếu kết quả cuối cùng là biểu thức toán học hoặc phương trình, BẮT BUỘC phải:
     + Giữ nguyên dạng PHÂN SỐ bằng `sp.nsimplify()` hoặc `sp.Rational()` để tránh làm tròn và tránh sai số
     + Chuẩn hóa dạng hiển thị của biểu thức bằng `sp.expand()` và sắp xếp theo thứ tự chuẩn (bậc cao → thấp) để dễ so sánh với đáp án
   - Nếu bài toán có nhiều bước: viết code mạch lạc, giải từng bước hợp lý.
   - Nếu bài toán có nhiều trường hợp: phải trả về tất cả các kết quả và các kết quả phụ quan trọng hỗ trợ việc tra cứu sau này.
   - Phải gán kết quả cuối cùng vào biến result kèm theo kết luận và đơn vị đo (nếu có) để câu trả lời có ý nghĩa.

3. QUAN TRỌNG - Xử lý bài toán lý thuyết (symbolic):
   - Nếu đề bài chỉ dùng ký hiệu (m, k, A, v₀, ω...) KHÔNG CHO GIÁ TRỊ CỤ THỂ và đáp án chứa công thức/biểu thức toán học thì Đây là BÀI TOÁN LÝ THUYẾT
   - KHÔNG ĐƯỢC gán giá trị cụ thể cho các biến ký hiệu (Ví dụ: KHÔNG gán kiểu m=1, k=1, A=1)
   - BẮT BUỘC phải phân tích biểu thức toán học theo lý thuyết

ĐẦU RA:
```python
"""
    log(f"[Math] Step 1: Generating code...")
    code_response = client.generate_math_code(code_gen_prompt)
    
    code = extract_code_block(code_response)
    if not code:
        code = code_response
    
    log(f"[Math] Generated code: {code}")
    
    # Self-Correction Loop: up to 4 retries (increased from 2 for better success rate)
    log(f"[Math] Step 2: Executing code with self-correction...")
    execution_result = None
    max_retries = 4  # Increased from 2 to 4 for better self-correction
    
    for attempt in range(max_retries + 1):
        try:
            # Pre-execution validation: Check for placeholder code
            if _is_placeholder_code(code):
                log(f"[Math] Attempt {attempt + 1}: Placeholder code detected, requesting complete code...")
                raise RuntimeError("Invalid code!")
            
            # Pre-execution validation: Check syntax before running
            is_valid, syntax_error = _validate_code_syntax(code)
            if not is_valid:
                log(f"[Math] Attempt {attempt + 1}: Syntax error: {syntax_error}")
                raise SyntaxError(f"SyntaxError: {syntax_error}")
            
            log(f"[Math] Attempt {attempt + 1}/{max_retries + 1}: Executing...")
            execution_result = python_executor.run(code)
            execution_result = execution_result.strip() if execution_result else "No output"
            
            # Check if result contains error messages (even if no exception thrown)
            # Comprehensive list of Python built-in exceptions
            error_indicators = [
                'Error', 'Exception', 'Traceback', 
                'NameError', 'ValueError', 'TypeError', 'SyntaxError', 'IndentationError',
                'AttributeError', 'KeyError', 'IndexError', 'ZeroDivisionError',
                'ImportError', 'ModuleNotFoundError', 'OverflowError', 'RecursionError',
                'AssertionError', 'SystemError', 'UnboundLocalError', 'NotImplementedError',
                'TimeoutError', 'FloatingPointError', 'ReferenceError', 'EOFError', 'StopIteration',
                'ArithmeticError', 'LookupError', 'RuntimeError', 'MemoryError', 'BufferError',
                'TabError', 'UnicodeError', 'UnicodeEncodeError', 'UnicodeDecodeError',
                'OSError', 'FileNotFoundError', 'PermissionError', 'ConnectionError', 'BlockingIOError'
            ]
            
            # Case-insensitive check
            result_lower = execution_result.lower()
            has_error = any(indicator.lower() in result_lower for indicator in error_indicators)
            
            if has_error:
                log(f"[Math] ✗ Error detected in output: {execution_result[:100]}")
                raise RuntimeError(f"Execution produced error: {execution_result}")
            
            log(f"[Math] ✓ Success: {execution_result[:100]}")
            break  # Success, exit loop
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            log(f"[Math] ✗ Error on attempt {attempt + 1}: {error_msg}")
            
            if attempt < max_retries:
                # Attempt to fix the code
                log(f"[Math] Attempting self-correction...")
                fix_prompt = f"""Code Python trước đó đã gặp lỗi khi thực thi. Hãy sửa lỗi và trả về code đã được sửa.

CODE GỐC:
```python
{code}
```

LỖI:
{error_msg}

YÊU CẦU:
1. Phân tích lỗi và xác định nguyên nhân
2. Sửa code để tránh lỗi này, luôn tuân theo quy tắc "Low code change".
3. Đảm bảo logic giải bài toán vẫn đúng
4. Chỉ trả về code Python đã sửa, không giải thích
5. Nếu lỗi là thiếu thư viện (no module named ...) thì KHÔNG ĐƯỢC PHÉP SỬ DỤNG THƯ VIỆN ĐÓ NỮA.

ĐỀ BÀI GỐC:
{question}

CÁC ĐÁP ÁN:
{choices_str}

CODE ĐÃ SỬA:
```python
"""             
                try:
                    fixed_response = client.generate_math_code(fix_prompt)
                    fixed_code = extract_code_block(fixed_response)
                    if fixed_code:
                        code = fixed_code
                        log(f"[Math] Generated fixed code: {code}")
                    else:
                        log(f"[Math] Failed to extract fixed code, using original")
                except Exception as fix_error:
                    log(f"[Math] Failed to generate fix: {fix_error}")
            else:
                # Max retries reached
                execution_result = f"Error after {max_retries + 1} attempts: {error_msg}"
                log(f"[Math] Max retries reached. Final error: {error_msg}")
    
    reasoning_prompt = f"""
Bạn là một trợ lý toán học thông minh có khả năng suy luận và dùng KẾT QUẢ TÍNH TOÁN làm tài liệu tham khảo để chọn đáp án chính xác nhất cho câu hỏi trắc nghiệm.

Kết quả tính toán: {execution_result}

Câu hỏi: {question}

Các đáp án:
{choices_str}

QUY TẮC:
1) ƯU TIÊN KẾT QUẢ THỰC THI:
   - Nếu kết quả tính toán chứa giá trị hữu ích (số, danh sách, chuỗi, boolean, dict với kết quả), hãy dùng nó làm bằng chứng chính để so khớp với các phương án trong `choices_str`.
   - Nếu kết quả tính toán là danh sách/tuple: kiểm tra tính tồn tại/đếm/độ dài/element-wise so với lựa chọn.
   - Nếu kết quả tính toán là boolean/True/False: so khớp với các phương án tương đương.
   - Đối với các bài toán mà nhân thêm bội số không làm thay đổi kết quả (vector pháp tuyến, vector chỉ phương,…), hãy xét các dạng tương đương.
   - Nếu một phương án KHỚP RÕ RÀNG với kết quả tính toán → chọn phương án đó.

   - PHẢI chuyển mọi phân số, biểu thức, phương trình trong các lựa chọn và trong kết quả tính toán về dạng số hoặc hệ số số học để so sánh (không so chuỗi).
   - PHẢI chuẩn hoá biểu thức toán (hàm số, phương trình, đa thức, vector…) về dạng giá trị toán học tương đương trước khi so khớp.
   - CHỈ so sánh theo giá trị toán học (chấp nhận sai số ±1e-9), không so sánh theo hình thức trình bày.

2) XỬ LÝ KẾT QUẢ KHÔNG RÕ / LỖI:
   - Nếu kết quả tính toán báo lỗi, rỗng, None, hoặc không parse được thì chuyển sang bước 3.
   - Nếu có nhiều kết quả mâu thuẫn: ưu tiên kết quả hợp lý nhất hoặc khớp toán học nhiều nhất.

3) FALLBACK — DÙNG KIẾN THỨC CỦA BẢN THÂN VÀ SUY LUẬN:
   - Khi không có hoặc không dùng được kết quả tính toán, dùng kiến thức sẵn có để suy luận đáp án hợp lý nhất.
   - Không được bịa đặt.

4) XỬ LÝ TRƯỜNG HỢP NHIỀU PHƯƠNG ÁN GẦN GIỐNG:
   - Ưu tiên phương án khớp toán học cao nhất.

5) CHUẨN HOÁ & SO KHỚP:
   - Chấp nhận các dạng tương đương: phân số ≈ số thập phân ≈ phần trăm.
   - Nhớ đổi về cùng dạng số.
   - Luôn so sánh theo giá trị toán học.

CHỈ TRẢ VỀ MỘT CHỮ CÁI ĐỨNG TRƯỚC CÂU TRẢ LỜI ĐÚNG — KHÔNG GIẢI THÍCH.
"""
    # Check if we should use CoT fallback (all executions failed)
    if execution_result and "Error after" in execution_result:
        log(f"[Math] Code execution completely failed, using CoT fallback")
        answer = _fallback_text_reasoning(question, choices)
        state["answer"] = answer
        state["reasoning"] = f"CoT fallback: code execution failed"
        log(f"[Math] Final answer (CoT): {state['answer']}")
        return state

    log(f"[Math] Step 3: Selecting answer...")
    answer_response = client.select_math_answer(reasoning_prompt)
    
    # Use robust answer extraction
    num_choices = len(choices)
    answer = extract_and_normalize(answer_response, num_choices, state.get("qid"), default="A")
    state["answer"] = answer
    
    state["reasoning"] = f"Code executed. Result: {execution_result[:100] if execution_result else 'None'}"
    log(f"[Math] Final answer: {state['answer']}")
    
    return state