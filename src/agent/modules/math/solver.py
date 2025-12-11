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
    
    code_gen_prompt = f"""
Bạn là chuyên gia giải toán bằng cách viết code Python chuyên nghiệp có độ chính xác tuyệt đối.
Nhiệm vụ: đọc hiểu đề bài và các đáp án được cung cấp để lựa chọn, phân tích yêu cầu, sau đó sinh ra đoạn code Python hoàn chỉnh để giải bài toán.

YÊU CẦU BẮT BUỘC:

1. Code phải:
   - Bắt buộc phải import đầy đủ thư viện cần dùng (nếu có) để tránh lỗi 
   - Bắt buộc phải định nghĩa đầy đủ các biến cần thiết (nếu có) để tránh lỗi
   - Chạy được ngay lập tức.
   - Không lỗi cú pháp.
   - Không lỗi runtime có thể dự đoán trước.
   - Đúng với yêu cầu của đề bài.

2. Hạn chế:
   - Chỉ sử dụng thư viện chuẩn Python: `math`, `statistics`, `fractions`, `decimal`, `itertools`, `collections`, `functools`, `heapq`, `bisect`, `random`, `re`, `string`.
   - Không dùng bất kỳ thư viện bên ngoài nào khác.

3. Định dạng đầu ra:
   - Trả về **duy nhất** một khối mã Python.
   - Không thêm bình luận, không giải thích, không văn bản thừa.

4. Quy tắc bắt buộc:
   - Code phải ngắn gọn, sạch sẽ, rõ ràng, không lặp lại vô nghĩa.
   - Nếu mà đoạn code có phép chia (tính thương) thì trong quá trình tính toán phải sử dụng kết quả ở dạng phân số, tránh đổi sang số thập phân để tránh sai số.
   - Nếu bài toán có nhiều bước: viết code mạch lạc, giải từng bước hợp lý.
   - Phải gán kết quả cuối cùng vào biến result kèm theo kết luận và đơn vị đo (nếu có) để câu trả lời có ý nghĩa.

QUY TRÌNH SUY LUẬN:

Trước khi sinh code, hãy tự động thực hiện 4 bước nội bộ:
1) Hiểu đề bài và các đáp án được cung cấp để lựa chọn: xác định đầu vào, đầu ra, công thức, các trường hợp biên.  
2) Chọn phương pháp giải phù hợp.
3) Xây dựng thuật toán rõ ràng.  
4) Viết code Python tối ưu và đúng chuẩn.

Nhưng chỉ trả về KHỐI CODE cuối cùng.
ĐỀ BÀI:
{question}

CÁC ĐÁP ÁN:
{choices_str}

ĐẦU RA:
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
    
    reasoning_prompt = f"""
Bạn là một trợ lý thông minh có khả năng suy luận và dùng KẾT QUẢ THỰC THI (nếu có) để chọn đáp án chính xác nhất.
QUY TẮC (thực hiện theo thứ tự):

1) ƯU TIÊN KẾT QUẢ THỰC THI:
   - Nếu `execution_result` chứa giá trị hữu ích (số, danh sách, chuỗi, boolean, dict với kết quả), hãy dùng nó làm bằng chứng chính để so khớp với các phương án trong `choices_str`.
   - Nếu `execution_result` là danh sách/tuple: kiểm tra tính tồn tại/đếm/độ dài/element-wise so với lựa chọn (ví dụ: "Có 3" ⇔ len(...) == 3).
   - Nếu `execution_result` là boolean/True/False: so khớp với các phương án có nội dung tương đương ("đúng/sai", "yes/no", v.v.).
   - Nếu một phương án KHỚP RÕ RÀNG với `execution_result` → chọn phương án đó.

2) XỬ LÝ KẾT QUẢ KHÔNG RÕ / LỖI:
   - Nếu `execution_result` rõ ràng báo lỗi, rỗng, None, hoặc không parse được thành dạng hữu ích thì chuyển sang bước 3 (dùng kiến thức của bản thân).
   - Nếu `execution_result` cho nhiều giá trị mâu thuẫn giữa các phương pháp thì ưu tiên kết quả cuối cùng (latest) nếu có đánh dấu thời gian; nếu không có thì thực hiện quy tắc "mức độ phù hợp" (bước 4).

3) FALLBACK — DÙNG KIẾN THỨC CỦA BẢN THÂN VÀ SUY LUẬN:
   - Khi không có hoặc không dùng được `execution_result`, dùng kiến thức sẵn có và logic để suy luận đáp án hợp lý nhất.
   - Không được bịa đặt thông tin — nếu không thể xác định rõ, vẫn cố chọn phương án có chứng cứ/logic mạnh nhất.

4) XỬ LÝ TRƯỜNG HỢP NHIỀU PHƯƠNG ÁN TƯƠNG ĐỒNG:
   - Nếu nhiều phương án đều khớp với `execution_result` ở cùng mức độ → chọn phương án có **mức khớp chuỗi tốt nhất** (exact string match → chứa kết quả → tương đương ngôn ngữ).

5) CHUẨN HOÁ VÀ SO KHỚP:
   - Chấp nhận biểu diễn số khác nhau nhưng bằng giá trị (ví dụ "0.5" = "1/2" = "50%").
   - So sánh ngày/thời gian ở dạng năm/tháng/ngày nếu thấy mẫu tương ứng.
   - Nhớ đổi về cùng đơn vị đo (nếu có)

ĐẦU VÀO:
Câu hỏi: {question}

Các đáp án:
{choices_str}

Kết quả tính toán: {execution_result}

CHỈ TRẢ VỀ MỘT CHỮ CÁI ĐỨNG TRƯỚC CÂU TRẢ LỜI ĐÚNG — KHÔNG GIẢI THÍCH.
"""


    log(f"[Math] Step 3: Selecting answer...")
    answer_response = client.select_math_answer(reasoning_prompt)
    match = re.match(r'([A-Z])', answer_response)
    if match:
        state["answer"] = match.group(1)
    else:
        state["answer"] = "B"
    
    state["reasoning"] = f"Code executed. Result: {execution_result[:100]}"
    log(f"[Math] Final answer: {state['answer']}")
    
    return state