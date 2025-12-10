from ....client import client
from ...state import AgentState
from ....logger import log
import re

class VectorDBPlaceholder:
    def search(self, query_embedding, top_k=5):
        return []

vector_db = VectorDBPlaceholder()

def rag_solver_node(state: AgentState) -> AgentState:
    """RAG solver using vnpt_large + embedding."""
    question = state["question"]
    choices = state["choices"]
    choices_str = "\n".join(choices)
    
    # Placeholder for vector search
    context = ""
    
    if context:
        prompt = f"""Dựa vào thông tin sau và kiến thức của bạn, hãy trả lời câu hỏi.

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Các đáp án:
{choices_str}

CHỈ TRẢ LỜI MỘT CHỮ CÁI (A, B, C, D,...) KHÔNG GIẢI THÍCH"""
    else:
        prompt = f"""Hãy trả lời câu hỏi sau dựa trên kiến thức của bạn.

Câu hỏi: {question}

Các đáp án:
{choices_str}

CHỈ TRẢ LỜI MỘT CHỮ CÁI (A, B, C, D,...) KHÔNG GIẢI THÍCH"""
    
    log(f"[RAG] Processing (context_len={len(context)})...")
    response = client.generate_rag_answer(prompt)
    
    answer = response.strip().upper()
    for char in answer:
        if char in 'ABCDEFGHIJ':
            state["answer"] = char
            break
    else:
        state["answer"] = "A"
    
    state["reasoning"] = f"RAG answer: {response[:50]}"
    state["context"] = context
    log(f"[RAG] Answer: {state['answer']}")
    
    return state
