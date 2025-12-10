from ....client import client
from ...state import AgentState
from ....logger import log
import re

def reading_solver_node(state: AgentState) -> AgentState:
    """Reading comprehension solver using vnpt_large."""
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    prompt = f"""Bạn là chuyên gia chuyên giải bài đọc hiểu nhiều lựa chọn (MCQ Reading Comprehension). 
Nhiệm vụ của bạn: đọc văn bản, đọc câu hỏi, phân tích tất cả đáp án và chọn DUY NHẤT một đáp án đúng nhất đối với văn bản cung cấp.

I. NGUYÊN TẮC CỐT LÕI
1) ONLY-IN-TEXT
   - Chỉ dùng thông tin có trong văn bản. Kết hợp với kiến thức có sẵn của bạn để phối hợp làm bài.
   - Không phỏng đoán, không suy diễn vượt ngoài văn bản.
2) TEXTUAL SUPPORT PRIORITY
   - Đáp án đúng phải được văn bản hỗ trợ trực tiếp hoặc gián tiếp nhưng rõ ràng.
   - “Rõ ràng” tức là có câu, cụm từ, lập luận hoặc bằng chứng trong văn bản tương ứng với nội dung đáp án.
3) BEST-SUPPORTED RULE
   - Nếu có nhiều đáp án có vẻ hợp lý, chọn đáp án có mức độ hỗ trợ mạnh nhất trong văn bản.
   - Không chọn đáp án chỉ “nghe hợp lý”.
4) NO CONTRADICTION
   - Nếu đáp án bị văn bản phủ định hoặc mâu thuẫn → loại ngay.

II. QUY TRÌNH PHÂN TÍCH 
Bước 1 — Xác định văn bản:  
- Lấy toàn bộ nội dung “Đoạn văn:", “Đoạn thông tin:” làm nguồn dữ liệu.
Bước 2 — Rút trích thông tin cốt lõi:  
- Xác định các thông tin quan trọng: quan hệ, hành động, lập luận ủng hộ / phản đối, mốc thời gian, định nghĩa, mô tả, so sánh.
Bước 3 — Phân tích câu hỏi:  
- Xác định dạng câu hỏi: yêu cầu thái độ, quan điểm, nguyên nhân, kết luận, thực tế, sự kiện, vai trò, mô tả đúng/sai.
Bước 4 — Đối chiếu từng đáp án:  
- Kiểm tra xem đáp án có được văn bản hỗ trợ hay không.  
- Đáp án mạnh nhất = được nhiều phần văn bản hỗ trợ nhất, hoặc được hỗ trợ trực tiếp rõ nhất.
Bước 5 — Loại trừ:  
- Loại đáp án bị mâu thuẫn với văn bản.  
- Loại đáp án chứa điều không được nhắc trong văn bản.
Bước 6 — Kết luận:  
- Nếu một đáp án vượt trội → chọn đáp án đó.  

III. QUY TẮC XỬ LÝ NHỮNG CÂU KHÓ (BẮT BUỘC)
1) QUAN ĐIỂM TRÁI CHIỀU  
   - Nếu văn bản nêu cả ý kiến ủng hộ và phản đối → xác định câu hỏi đang hỏi “theo ngữ cảnh”, không theo phe.  
   - Chọn đáp án mô tả khách quan toàn văn bản, không thiên lệch.
2) THÔNG TIN NHIỀU ĐOẠN  
   - Kết hợp thông tin từ nhiều đoạn nếu cần, nhưng vẫn phải dựa hoàn toàn vào văn bản.
3) SUY LUẬN GIÁN TIẾP  
   - Chỉ chấp nhận suy luận gián tiếp nếu văn bản cung cấp đủ dữ kiện dẫn tới kết luận đó.
4) CHỌN ĐÁP ÁN “VAI TRÒ / ẢNH HƯỞNG”  
   - Khi hỏi về “vai trò”, “ảnh hưởng”, “bản chất”, “ý nghĩa” → chọn đáp án phù hợp với toàn bộ mô tả về chủ thể đó trong văn bản.

Dưới đây là câu hỏi và các đáp án:
Câu hỏi:
{question}

Các đáp án:
{choices_str}

CHỈ TRẢ LỜI BẰNG CHỮ CÁI ĐỨNG TRƯỚC CÂU TRẢ LỜI ĐÚNG, KHÔNG GIẢI THÍCH"""
    
    log(f"[Reading] Processing...")
    response = client.generate_reading_answer(prompt)
    match = re.match(r'([A-Z])', response)
    if match:
        state["answer"] = match.group(1)
    else:
        state["answer"] = "A"
    
    state["reasoning"] = f"Reading comprehension: {response[:50]}"
    log(f"[Reading] Answer: {state['answer']}")
    
    return state