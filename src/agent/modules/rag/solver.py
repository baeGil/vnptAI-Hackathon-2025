from ....client import client
from ...state import AgentState
import re

# Placeholder for vector database - sẽ được implement chi tiết sau
class VectorDBPlaceholder:
    def search(self, query_embedding, top_k=5):
        """Placeholder: returns empty context."""
        return []

vector_db = VectorDBPlaceholder()

def rag_solver_node(state: AgentState) -> AgentState:
    """
    RAG (Retrieval-Augmented Generation) solver.
    Uses:
    - client.get_embedding() - vnpt_embedding: Embed query
    - client.generate_rag_answer() - vnpt_large: Generate answer
    
    Input: Câu hỏi + đáp án đã format (A. xxx, B. xxx,...)
    Output: Chỉ trả về một chữ cái (A, B, C, D,...)
    """
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    # Step 1: Get embedding and search (placeholder)
    # query_embedding = client.get_embedding(question)
    # retrieved_docs = vector_db.search(query_embedding, top_k=5)
    # context = "\n".join([doc['content'] for doc in retrieved_docs])
    context = ""
    
    # Step 2: Generate answer
    if context:
        prompt = f"""Bạn là trợ lý tri thức thông minh.

**Thông tin tham khảo:**
{context}

**Câu hỏi:** {question}

**Các đáp án:**
{choices_str}

**Yêu cầu:**
1. Sử dụng thông tin tham khảo ở trên
2. Kết hợp với kiến thức của bạn
3. Chọn đáp án chính xác nhất

**CHỈ TRẢ LỜI MỘT CHỮ CÁI (A, B, C, D, E,...) KHÔNG GIẢI THÍCH**"""
    else:
        # Zero-context: rely on model's knowledge
        prompt = f"""Bạn là trợ lý tri thức thông minh với kiến thức rộng về lịch sử, địa lý, văn hóa, chính trị, khoa học, và các lĩnh vực khác.

**Câu hỏi:** {question}

**Các đáp án:**
{choices_str}

**Yêu cầu:**
1. Suy nghĩ cẩn thận về câu hỏi
2. Phân tích từng đáp án
3. Chọn đáp án chính xác nhất dựa trên kiến thức của bạn

**CHỈ TRẢ LỜI MỘT CHỮ CÁI (A, B, C, D, E,...) KHÔNG GIẢI THÍCH**"""
    
    print(f"[RAG] Processing using generate_rag_answer() (context_len={len(context)})...")
    response = client.generate_rag_answer(prompt)
    
    # Extract answer letter
    answer = response.strip().upper()
    for char in answer:
        if char in 'ABCDEFGHIJ':
            state["answer"] = char
            break
    else:
        state["answer"] = "A"
    
    state["reasoning"] = f"RAG answer: {response[:100]}"
    state["context"] = context
    print(f"[RAG] Answer: {state['answer']}")
    
    return state