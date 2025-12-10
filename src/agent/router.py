from ..client import client
from .state import AgentState
from ..logger import log
import json
import re

def router_node(state: AgentState) -> AgentState:
    """
    Router node using vnpt_hackathon_small for classification.
    Returns JSON: {'type': ..., 'toxic_detected': ...}
    If toxic detected, sets answer directly without going to toxic module.
    """
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    classification_prompt = f"""Bạn là một trợ lý phân loại câu hỏi. Nhiệm vụ của bạn là phân loại câu hỏi sau vào MỘT trong các nhóm TOXIC, MATH, READING, RAG:
=========================
ĐỊNH NGHĨA NHÓM
=========================
1. TOXIC
   - Một hoặc nhiều đáp án chứa nội dung kiểu từ chối trả lời hoặc nằm ngoài phạm vi trả lời.
   - Dấu hiệu: “tôi không thể trả lời”, “tôi không biết”, “không thể cung cấp”, “không thể chia sẻ”, “nằm ngoài phạm vi”, “bất hợp pháp”, "không thể hỗ trợ", "nằm ngoài phạm vi trả lời", "nằm ngoài tầm hiểu biết", etc.
   - Quy tắc ưu tiên: Nếu đáp án chứa nội dung từ chối → gán TOXIC ngay lập tức.
2. MATH
   - Bài toán có dấu hiệu cần THỰC HIỆN TÍNH TOÁN, SUY LUẬN, LẬP TRÌNH hoặc THAO TÁC LOGIC.
   - Có thể giải bằng Python hoặc bằng thao tác tính toán tuần tự.
   - Câu trả lời KHÔNG THỂ có được chỉ bằng việc nhớ định nghĩa.
   - **TIÊU CHÍ BẮT BUỘC ĐỂ LÀ MATH:**
        (a) Phải có phép tính cần thực hiện (nhân, chia, cộng, trừ, đếm, liệt kê, mô phỏng, tính toán, ước lượng, đạo hàm, vi phân, tích phân, xác suất, tổ hợp, thống kê, ...) để tìm kết quả.
        (b) Người giải phải xử lý thông tin từ đề, không phải trích từ kiến thức có sẵn.
3. READING
   - Câu hỏi cung cấp một đoạn văn hoặc một đoạn dữ liệu rõ ràng (ví dụ: "Đoạn thông tin:", "Đoạn văn:", "Cho đoạn văn sau:", "Cho đoạn thông tin:",...)
   - Câu trả lời phải dựa trực tiếp vào nội dung đoạn văn đó.
   - Nếu không có đoạn văn → không phải READING.
4. RAG
   - Câu hỏi yêu cầu kiến thức bên ngoài hoặc khái niệm lý thuyết.
   - Bao gồm: kiến thức tổng hợp, lịch sử, địa lý, văn hóa, chính trị, sinh học, công thức, quy tắc, định nghĩa, luật, dữ kiện thế giới,...
   - **LƯU Ý QUAN TRỌNG:**  
       (a) Nếu câu hỏi CHỈ hỏi về công thức lý thuyết, định lý, nguyên lý, quy tắc vật lý/hóa học,...:
          DÙ CÓ các biểu thức toán trong đáp án thì vẫn là RAG, vì Math đòi hỏi thực hiện phép tính nào đó.
       (b) Nếu không thể phân loại vào MATH< READING hoặc TOXIC thì sẽ là RAG.
Câu hỏi:
{question}

Các đáp án:
{choices_str}

Trả về JSON theo định dạng sau (không giải thích thêm):
{{"type": "TOXIC" hoặc "MATH" hoặc "READING" hoặc "RAG", "toxic_detected": null nếu type không phải TOXIC, hoặc "A" hoặc "B" hoặc "C"... nếu type là TOXIC (đáp án chứa nội dung từ chối, chữ cái đầu đứng trước đáp án)}}"""

    log(f"[Router] Classifying question...")
    response = client.classify_router(classification_prompt)
    
    # Parse JSON response
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response.strip())
        
        q_type = result.get("type", "RAG").upper()
        toxic_detected = result.get("toxic_detected")
        
    except (json.JSONDecodeError, Exception) as e:
        log(f"[Router] JSON parse error: {e}, response: {response}")
        # Fallback to simple parsing
        response_upper = response.strip().upper()
        if "TOXIC" in response_upper:
            q_type = "TOXIC"
            toxic_detected = None
        elif "MATH" in response_upper:
            q_type = "MATH"
            toxic_detected = None
        elif "READING" in response_upper:
            q_type = "READING"
            toxic_detected = None
        else:
            q_type = "RAG"
            toxic_detected = None
    
    # Set category
    state["category"] = q_type.lower()
    
    # If toxic detected with answer, set answer directly
    if q_type == "TOXIC" and toxic_detected:
        toxic_answer = str(toxic_detected).strip().upper()
        
        # Tìm kiếm ký tự A-Z đầu tiên
        match = re.match(r'([A-Z])', toxic_answer)
        if match:
            toxic_char = match.group(1)
            
            state["answer"] = toxic_char
            state["reasoning"] = f"Toxic detected by router: {toxic_char}"
            log(f"[Router] -> TOXIC (answer: {toxic_char} detected directly)")
        else:
            log(f"[Router] -> TOXIC (invalid answer: {toxic_detected}, will use toxic module)")
    else:
        log(f"[Router] -> {q_type}")
    
    return state