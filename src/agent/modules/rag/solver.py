#!/usr/bin/env python3
"""
Optimized RAG Solver - Semantic Vector Search.
Clean implementation: Query → HNSW Search → Context → Answer.
"""
import os
import sys
import re
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.client import client
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from ....logger import log

# Config
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vnpt_rag_v2"

# Search Parameters
TOP_K = 10                   # Number of results to retrieve (more for reranking)
RERANK_TOP_K = 5             # Number of docs after reranking
MIN_SCORE = 0.35             # Minimum cosine similarity threshold
MAX_CONTEXT_CHARS = 4000     # Max context length to avoid token overflow

# HNSW Search Parameters (higher ef = better recall, slower)
HNSW_EF = 128                # Search accuracy

# Reranker
def _rerank_documents(query: str, docs: List[dict], top_k: int = RERANK_TOP_K) -> List[dict]:
    """
    Rerank retrieved documents using small LLM for better precision.
    
    Args:
        query: The user question
        docs: List of retrieved documents
        top_k: Number of top documents to return after reranking
    
    Returns:
        List of reranked documents (top_k most relevant)
    """
    if len(docs) <= top_k:
        return docs
    
    # Build document list for reranking prompt
    doc_list = ""
    for i, doc in enumerate(docs):
        content_preview = doc.get("content", "")[:500].replace("\n", " ")
        doc_list += f"[{i}] {content_preview}...\n\n"
    
    try:
        response = client.rerank_documents(query, doc_list, top_k)
        # Parse selected IDs from response
        import re
        selected_ids = []
        numbers = re.findall(r'\d+', response)
        for num_str in numbers:
            idx = int(num_str)
            if 0 <= idx < len(docs) and idx not in selected_ids:
                selected_ids.append(idx)
                if len(selected_ids) >= top_k:
                    break
        
        if selected_ids:
            reranked = [docs[i] for i in selected_ids]
            return reranked
        
        print("[RAG] Rerank parsing failed, using first top_k docs")
        return docs[:top_k]
        
    except Exception as e:
        print(f"[RAG] Reranking failed: {e}. Using first {top_k} docs.")
        return docs[:top_k]

# qdrant client
_qdrant: Optional[QdrantClient] = None

def get_qdrant() -> Optional[QdrantClient]:
    """Lazy initialization of Qdrant client."""
    global _qdrant
    if _qdrant is None:
        try:
            _qdrant = QdrantClient(url=QDRANT_URL, timeout=10)
        except Exception as e:
            print(f"[RAG] Qdrant connection failed: {e}")
            return None
    return _qdrant

# Vector search
def vector_search(query: str, top_k: int = TOP_K) -> List[dict]:
    """
    Perform HNSW vector search on Qdrant.
    
    Args:
        query: Combined question + options text
        top_k: Number of results to retrieve
        
    Returns:
        List of {content, score, metadata} dicts
    """
    qdrant = get_qdrant()
    if not qdrant:
        return []
    
    # Embed query using VNPT API
    query_vector = client.get_embedding(query)
    if not query_vector:
        print("[RAG] Embedding failed")
        return []
    
    # HNSW Search with tuned parameters
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=MIN_SCORE,
            search_params=SearchParams(
                hnsw_ef=HNSW_EF,      # Higher = better recall
                exact=False           # Use HNSW index (not brute force)
            ),
            with_payload=True,
            with_vectors=False
        ).points
    except Exception as e:
        print(f"[RAG] Search error: {e}")
        return []
    
    # Convert to simple dict format
    docs = []
    for hit in results:
        docs.append({
            "content": hit.payload.get("content", ""),
            "score": hit.score,
            "domain": hit.payload.get("domain", ""),
            "doc_title": hit.payload.get("doc_title", ""),
            "article_num": hit.payload.get("article_num", ""),
            "chapter": hit.payload.get("chapter", "")
        })
    
    return docs

# Context builder
def build_context(docs: List[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Build context string from retrieved documents.
    Features: Deduplication, length control, domain-aware formatting.
    """
    if not docs:
        return ""
    
    contexts = []
    seen = set()
    total_len = 0
    
    for doc in docs:
        content = doc["content"].strip()
        
        # Deduplicate by first 150 chars
        key = content[:150]
        if key in seen:
            continue
        seen.add(key)
        
        # Format header based on domain
        domain = doc.get("domain", "")
        if domain == "legal":
            parts = [p for p in [doc.get("doc_title"), doc.get("chapter"), doc.get("article_num")] if p]
            header = f"[{parts}]" if parts else "[Văn bản pháp luật]"
        elif domain == "mcq":
            header = f"[{doc.get('doc_title', 'Câu hỏi tham khảo')}]"
        else:
            header = f"[{doc.get('doc_title', 'Tài liệu')}]"
        
        # Check length
        entry = f"{header}\n{content}"
        if total_len + len(entry) > max_chars:
            remaining = max_chars - total_len
            if remaining < 200:
                break
            entry = f"{header}\n{content[:remaining-len(header)-10]}..."
        
        contexts.append(entry)
        total_len += len(entry)
        
        if total_len >= max_chars:
            break
    
    return "\n\n---\n\n".join(contexts)

# Answer generator
def generate_answer(question: str, choices: List[str], context: str) -> str:
    """
    Generate answer using LLM with retrieved context.
    Returns single letter (A/B/C/D).
    """
    from ....answer import extract_and_normalize
    
    choices_text = "\n".join(choices)
    prompt = f"""Bạn là trợ lý AI trung thực. Nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm CHỈ DỰA TRÊN đoạn văn bản được cung cấp.

Văn bản:
{context}

Quy tắc bắt buộc:
1. Nếu văn bản chứa thông tin trả lời: Hãy suy luận logic và chọn ra đáp án đúng nhất
2. Nếu văn bản KHÔNG chứa thông tin liên quan:
- Hãy sử dụng kiến thức của bạn để suy luận và chọn đáp án mà bạn cho là hợp lý nhất về mặt logic chung (common sense).

CÂU HỎI: {question}

CÁC LỰA CHỌN:
{choices_text}

Cuối cùng trả lời theo định dạng: Đáp án: X (X là một chữ cái)
"""

    try:
        response = client.generate_rag_answer(prompt)
        answer = extract_and_normalize(response, len(choices), default="A")
        return answer
    except Exception as e:
        print(f"[RAG] LLM error: {e}")
        return "A"

# Main RAG pipeline
def rag_search(question: str, choices: List[str]) -> tuple[str, List[dict]]:
    """
    Main RAG search function.
    
    Args:
        question: The question text
        choices: List of answer choices
        
    Returns:
        (context_string, list_of_docs)
    """
    # Build query = question + all options
    query = f"{question}\n" + "\n".join(choices)
    
    # Vector search
    docs = vector_search(query, top_k=TOP_K)
    
    # Log detailed results for monitoring
    try:
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "output", "inference_detail.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"QUERY: {question}\n")
            f.write(f"CHOICES: {choices}\n")
            f.write(f"FOUND: {len(docs)} docs\n")
            
            for idx, doc in enumerate(docs):
                title = doc.get("doc_title", "No Title")
                domain = doc.get("domain", "unknown")
                score = doc.get("score", 0.0)
                content_snippet = doc.get("content", "")[:200].replace("\n", " ")
                f.write(f"  [{idx+1}] Score: {score:.4f} | {domain.upper()} | {title}\n")
                f.write(f"      Context: {content_snippet}...\n")
                
            f.write(f"{'='*80}\n")
    except Exception as e:
        print(f"[RAG] Logging failed: {e}")
    
    if not docs:
        return "", []
    
    # Rerank documents using LLM
    reranked_docs = _rerank_documents(question, docs, top_k=RERANK_TOP_K)
    
    # Build context from reranked docs
    context = build_context(reranked_docs)
    
    return context, reranked_docs

def rag_solver_node(agent_state: dict) -> dict:
    """
    RAG solver node for agent graph.
    Pipeline: Query → Vector Search → Context → Answer
    Includes fallback to LLM knowledge when retrieval is weak.
    """
    question = agent_state["question"]
    choices = agent_state["choices"]
    
    # Step 1: Search
    context, docs = rag_search(question, choices)
    
    # Step 2: Check context quality and decide approach
    MIN_CONTEXT_THRESHOLD = 500  # Minimum chars for useful context
    use_fallback = not context or len(context) < MIN_CONTEXT_THRESHOLD
    
    if use_fallback:
        log(f"[RAG] Weak context ({len(context) if context else 0} chars), using LLM fallback")
        answer = _fallback_answer_with_llm(question, choices, context)
        agent_state["answer"] = answer
        agent_state["reasoning"] = f"RAG fallback (weak context): {len(docs)} docs"
        agent_state["context"] = context[:500] if context else ""
        return agent_state
    
    # Step 3: Generate answer with RAG context
    answer = generate_answer(question, choices, context)
    
    # Step 4: Update state
    agent_state["answer"] = answer
    agent_state["reasoning"] = f"RAG: {len(docs)} docs, {len(context)} chars"
    agent_state["context"] = context[:500]
    
    return agent_state


def _fallback_answer_with_llm(question: str, choices: list, weak_context: str = "") -> str:
    """
    Fallback: Use LLM's built-in knowledge when RAG retrieval is weak.
    """
    choices_str = "\n".join(choices)
    
    prompt = f"""Bạn là chuyên gia trả lời câu hỏi trắc nghiệm. Dựa vào kiến thức của bạn, hãy chọn đáp án đúng nhất.

Câu hỏi: {question}

Các đáp án:
{choices_str}

{f"Thông tin tham khảo (có thể không đầy đủ): {weak_context}" if weak_context else ""}

QUY TẮC:
1. Sử dụng kiến thức chuyên môn của bạn để phân tích câu hỏi
2. Xem xét từng đáp án và loại bỏ các đáp án sai
3. Chọn đáp án đúng nhất dựa trên logic và kiến thức
4. CHỈ TRẢ LỜI BẰNG MỘT CHỮ CÁI (A, B, C, D...) - KHÔNG GIẢI THÍCH
"""
    
    log(f"[RAG] Using LLM fallback for weak retrieval")
    response = client.generate_rag_answer(prompt)
    
    # Extract answer
    import re
    match = re.match(r'\s*([A-Z])', response)
    if match:
        return match.group(1)
    return "A"

# A simple test example
if __name__ == "__main__":
    
    test_q = "Thế nào là hợp đồng dân sự?"
    test_choices = [
        "A. Là sự thỏa thuận giữa các bên",
        "B. Là văn bản do nhà nước ban hành",
        "C. Là quyết định hành chính",
        "D. Là bản án của tòa án"
    ]
    
    print("Testing RAG Pipeline...")
    context, docs = rag_search(test_q, test_choices)
    print(f"\nFound {len(docs)} documents")
    print(f"Context preview:\n{context[:500]}...")