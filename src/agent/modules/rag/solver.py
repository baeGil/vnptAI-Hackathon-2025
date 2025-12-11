#!/usr/bin/env python3
"""
Advanced RAG Solver with Multi-Query, Re-ranking, and Context Optimization.
Techniques:
- Query Expansion (multiple perspectives)
- Reciprocal Rank Fusion (RRF)
- Context Deduplication & Compression
- Smart Citation Formatting
"""
import os
import sys
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.client import client, RateLimitException
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vnpt_rag"
TOP_K_PER_QUERY = 10  # Fetch more candidates for re-ranking
FINAL_TOP_K = 5  # Final context size after fusion
MIN_SCORE_THRESHOLD = 0.3  # Minimum similarity score

# Global Qdrant
_qdrant = None

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        try:
            _qdrant = QdrantClient(url=QDRANT_URL)
        except:
            return None
    return _qdrant

# MULTI-VECTOR SEARCH
# ============================================

def search_multi_query(queries: List[str], k: int = TOP_K_PER_QUERY) -> List[Tuple[ScoredPoint, str]]:
    """
    Perform search for multiple queries and collect all results.
    Returns: [(scored_point, query_source), ...]
    """
    qdrant = get_qdrant()
    if not qdrant:
        return []
    
    all_results = []
    
    for query_text in queries:
        try:
            # Embed query
            query_vector = client.get_embedding(query_text)
            if not query_vector:
                continue
            
            # Search
            results = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=k,
                with_payload=True,
                with_vectors=False,
                score_threshold=MIN_SCORE_THRESHOLD
            ).points
            
            # Tag with query source
            for hit in results:
                all_results.append((hit, query_text))
                
        except Exception as e:
            print(f"[Search Error] {e}")
            continue
    
    return all_results

# ============================================
# RECIPROCAL RANK FUSION (RRF)
# ============================================

def reciprocal_rank_fusion(results: List[Tuple[ScoredPoint, str]], k: int = 60) -> List[ScoredPoint]:
    """
    Combine results from multiple queries using RRF.
    RRF formula: score = sum(1 / (k + rank)) for each query where doc appears.
    
    This handles cases where:
    - Same document retrieved by different queries (boost)
    - Different ranking positions (weighted fusion)
    """
    # Group by document ID
    doc_scores = defaultdict(lambda: {'point': None, 'rrf_score': 0.0, 'sources': []})
    
    # Build rankings per query
    query_rankings = defaultdict(list)
    for point, query in results:
        query_rankings[query].append(point)
    
    # Calculate RRF scores
    for query, points in query_rankings.items():
        for rank, point in enumerate(points):
            doc_id = point.id
            rrf_score = 1.0 / (k + rank + 1)
            
            doc_scores[doc_id]['point'] = point
            doc_scores[doc_id]['rrf_score'] += rrf_score
            doc_scores[doc_id]['sources'].append(query)
    
    # Sort by RRF score
    ranked_docs = sorted(
        doc_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )
    
    return [item['point'] for item in ranked_docs]

# ============================================
# CONTEXT ASSEMBLY
# ============================================

def format_context(points: List[ScoredPoint], max_context_length: int = 6000) -> str:
    """
    Format retrieved documents into structured context.
    Features:
    - Deduplication
    - Smart citation
    - Length control
    """
    contexts = []
    seen_content = set()
    total_length = 0
    
    for point in points:
        payload = point.payload
        content = payload.get("content", "").strip()
        
        # Deduplicate by content hash
        content_hash = hash(content[:200])  # Hash first 200 chars
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        # Build citation header
        header = ""
        if payload.get("domain") == "legal":
            parts = []
            if payload.get("doc_title"):
                parts.append(payload["doc_title"])
            if payload.get("chapter"):
                parts.append(payload["chapter"])
            if payload.get("article_num"):
                parts.append(payload["article_num"])
            header = f"[📜 {' - '.join(parts)}]"
        else:
            header = f"[ {payload.get('doc_title', 'Tài liệu')}]"
        
        # Truncate if needed
        if total_length + len(content) > max_context_length:
            remaining = max_context_length - total_length
            if remaining < 200:  # Not enough space
                break
            content = content[:remaining] + "..."
        
        formatted = f"{header}\n{content}"
        contexts.append(formatted)
        total_length += len(formatted)
        
        if total_length >= max_context_length:
            break
    
    return "\n\n---\n\n".join(contexts)

# ============================================
# MAIN RAG PIPELINE
# ============================================

def advanced_rag_search(question: str, choices: List[str]) -> str:
    """
    Advanced RAG without query expansion.
    Uses: Multi-vector search (question + choices) + RRF fusion.
    """
    # Build comprehensive query from question and choices
    choices_text = "\n".join(choices)
    query = f"{question}\n{choices_text}"
    
    # Single query search (no expansion/rephrase)
    queries = [query]
    
    # Multi-Query Search (just one query now, but keeps infrastructure)
    all_results = search_multi_query(queries, k=TOP_K_PER_QUERY)
    
    if not all_results:
        print("[RAG] No results found")
        return ""
    
    # RRF Fusion (still useful even with single query for consistency)
    fused_results = reciprocal_rank_fusion(all_results)
    final_results = fused_results[:FINAL_TOP_K]
    
    print(f"[RAG] Retrieved {len(final_results)} documents")
    
    # Format Context
    context = format_context(final_results)
    
    return context

def rag_solver_node(agent_state):
    """
    RAG solver node for the main agent graph.
    Uses advanced retrieval (RRF, hybrid search) WITHOUT query expansion.
    """
    question = agent_state["question"]
    choices = agent_state["choices"]
    
    print(f"[RAG] Processing: {question[:60]}...")
    
    # Advanced search without expansion
    context = advanced_rag_search(question, choices)
    
    # Generate answer - ALWAYS use unified prompt
    choices_text = "\n".join(choices)
    
    # Unified prompt that combines context + internal knowledge
    context_section = f"""
TÀI LIỆU CUNG CẤP:
{context}
"""
    
    prompt = f"""Bạn là chuyên gia về suy luận và có kiến thức tổng quát.
Nhiệm vụ: Trả lời câu hỏi trắc nghiệm dưới đây dựa vào ngữ cảnh tài liệu được cung cấp:
NGỮ CẢNH TÀI LIỆU:
{context_section}

CÂU HỎI: {question}

CÁC LỰA CHỌN:
{choices_text}

HƯỚNG DẪN:
1. Đọc kỹ ngữ cảnh được cung cấp
2. Suy luận để tìm đáp án chính xác nhất
CHỈ TRẢ LỜI BẰNG MỘT CHỮ CÁI ĐỨNG TRƯỚC CÂU TRẢ LỜI ĐÚNG. KHÔNG GIẢI THÍCH."""
    
    try:
        response = client.generate_rag_answer(prompt)
        
        # Extract answer
        import re
        match = re.search(r'\b([A-Z])\b', response.strip().upper())
        answer = match.group(1) if match else "A"
        
        agent_state["answer"] = answer
        agent_state["reasoning"] = f"RAG: {len(context)} chars context"
        agent_state["context"] = context[:500] if context else ""
        
        print(f"[RAG] Answer: {answer}")
        
    except Exception as e:
        print(f"[RAG Error] {e}")
        agent_state["answer"] = "A"
        agent_state["reasoning"] = f"Error: {str(e)[:100]}"
    
    return agent_state