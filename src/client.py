import requests
from .config import config

class RateLimitException(Exception):
    """Exception raised when API rate limit (429) or quota exceeded (401) is hit."""
    pass

class VNPTClient:
    """
    VNPT AI API Client with methods for each task.
    Automatically detects rate limit (429/401) and raises RateLimitException.
    """
    
    def __init__(self):
        self.base_url = config.VNPT_API_BASE_URL
        
        # Headers for Small model
        self.small_headers = {
            'Authorization': config.VNPT_SMALL_API_KEY,
            'Token-id': config.VNPT_SMALL_TOKEN_ID,
            'Token-key': config.VNPT_SMALL_TOKEN_KEY,
            'Content-Type': 'application/json',
        }
        
        # Headers for Large model
        self.large_headers = {
            'Authorization': config.VNPT_LARGE_API_KEY,
            'Token-id': config.VNPT_LARGE_TOKEN_ID,
            'Token-key': config.VNPT_LARGE_TOKEN_KEY,
            'Content-Type': 'application/json',
        }
        
        # Headers for Embedding model
        self.embedding_headers = {
            'Authorization': config.VNPT_EMBEDDING_API_KEY,
            'Token-id': config.VNPT_EMBEDDING_TOKEN_ID,
            'Token-key': config.VNPT_EMBEDDING_TOKEN_KEY,
            'Content-Type': 'application/json',
        }

    def _check_rate_limit(self, response: requests.Response, method_name: str):
        """Check for rate limit errors and raise RateLimitException if detected."""
        if response.status_code in [429, 401]:
            try:
                detail = response.text
            except:
                detail = "No details"
            error_msg = f"[{method_name}] Rate Limit/Quota exceeded: HTTP {response.status_code} - {detail}"
            print(error_msg)
            
            # If 401, it might be fatal (Invalid Key), unless API uses 401 for Quota.
            # We raise the same exception but with more detail so user sees it in logs.
            raise RateLimitException(error_msg)
        response.raise_for_status()

    def _parse_response(self, response: requests.Response, method_name: str) -> str:
        """Safely parse API response content and log details on failure."""
        try:
            data = response.json()
            return data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as e:
            # Log the actual response body to debug 'KeyError: choices'
            print(f"[{method_name}] Invalid Response format: {e} | Body: {response.text[:500]}")
            return ""
        except Exception as e:
            print(f"[{method_name}] Parse Error: {e}")
            return ""

# 6 Different model configs for 6 specific tasks

    # Task1: ROUTER CLASSIFICATION (vnpt_hackathon_small)
    def classify_router(self, prompt: str) -> str:
        """Classify question using vnpt_hackathon_small."""
        url = f"{self.base_url}/vnptai-hackathon-small"
        payload = {
            'model': 'vnptai_hackathon_small',
            'messages': [{"role": "user", "content": prompt}],
            'max_completion_tokens': 20,
            'temperature': 0.0,
            'top_k': 1,
            'n': 1,
            'seed': 42,
            'response_format': {'type': 'json_object'},
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.small_headers, json=payload, timeout=120)
            self._check_rate_limit(response, "classify_router")
            return self._parse_response(response, "classify_router")
        except RateLimitException:
            raise  # Re-raise to be caught by predict.py
        except Exception as e:
            print(f"[classify_router] Error: {e}")
            return ""

    # Task 2: MATH CODE GENERATION (vnpt_hackathon_large)
    def generate_math_code(self, prompt: str) -> str:
        """Generate Python code using vnpt_hackathon_large."""
        url = f"{self.base_url}/vnptai-hackathon-large"
        payload = {
            'model': 'vnptai_hackathon_large',
            'messages': [
                {"role": "system", "content": "Bạn là chuyên gia lập trình code Python. Luôn trả về code trong markdown code block."},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': 4096,
            'temperature': 0.0,
            'top_p': 0.9,
            'top_k': 25,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=120)
            self._check_rate_limit(response, "generate_math_code")
            return self._parse_response(response, "generate_math_code")
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[generate_math_code] Error: {e}")
            return ""

    # Task 3: MATH ANSWER SELECTION (vnpt_hackathon_large)
    def select_math_answer(self, prompt: str) -> str:
        """Select final answer for math problem using vnpt_hackathon_large."""
        url = f"{self.base_url}/vnptai-hackathon-large"
        payload = {
            'model': 'vnptai_hackathon_large',
            'messages': [
                {"role": "system", "content": "Bạn là trợ lý chọn đáp án. Dựa vào kết quả tính toán, hãy chọn đáp án đúng nhất."},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': 10,
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 10,
            # 'presence_penalty': 0.0,
            # 'frequency_penalty': 0.0,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=120)
            self._check_rate_limit(response, "select_math_answer")
            return self._parse_response(response, "select_math_answer")
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[select_math_answer] Error: {e}")
            return ""

    # Task 4: RAG GENERATION (vnpt_hackathon_large)
    def generate_rag_answer(self, prompt: str) -> str:
        """Generate RAG answer using vnpt_hackathon_large."""
        url = f"{self.base_url}/vnptai-hackathon-large"
        payload = {
            'model': 'vnptai_hackathon_large',
            'messages': [
                {"role": "system", "content": "Bạn là trợ lý tri thức. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp và kiến thức của bạn."},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': 10,
            'temperature': 0.0,
            'top_p': 1.0,
            # 'top_k': 30,
            # 'presence_penalty': 0.0,
            # 'frequency_penalty': 0.0,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=120)
            self._check_rate_limit(response, "generate_rag_answer")
            return self._parse_response(response, "generate_rag_answer")
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[generate_rag_answer] Error: {e}")
            return ""

    # Task 5: READING COMPREHENSION (vnpt_hackathon_large)
    def generate_reading_answer(self, prompt: str) -> str:
        """Generate reading answer using vnpt_hackathon_large."""
        url = f"{self.base_url}/vnptai-hackathon-large"
        payload = {
            'model': 'vnptai_hackathon_large',
            'messages': [
                {"role": "system", "content": "Bạn là chuyên gia đọc hiểu văn bản. Hãy đọc kỹ đoạn văn và trả lời chính xác theo nội dung."},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': 10,
            'temperature': 0.0,
            'top_p': 1.0,
            # 'top_k': 30,
            # 'presence_penalty': 0.0,
            # 'frequency_penalty': 0.0,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=120)
            self._check_rate_limit(response, "generate_reading_answer")
            return self._parse_response(response, "generate_reading_answer")
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[generate_reading_answer] Error: {e}")
            return ""

    # Task 6: DOCUMENT RERANKING (vnpt_hackathon_small)
    def rerank_documents(self, query: str, doc_list: str, top_k: int = 3) -> str:
        """Rerank documents using vnpt_hackathon_small for better precision."""
        url = f"{self.base_url}/vnptai-hackathon-small"
        
        system_prompt = (
            "Bạn là chuyên gia đánh giá độ liên quan của văn bản. "
            "Nhiệm vụ: Chọn ra các đoạn văn bản LIÊN QUAN NHẤT với câu hỏi.\n"
            "Chỉ trả về danh sách các số ID (ví dụ: 0, 3, 5), không giải thích."
        )
        
        user_prompt = (
            f"Câu hỏi: {query}\n\n"
            f"Các đoạn văn bản:\n{doc_list}\n"
            f"Hãy chọn {top_k} đoạn văn bản LIÊN QUAN NHẤT với câu hỏi. "
            f"Trả về danh sách ID, cách nhau bởi dấu phẩy."
        )
        
        payload = {
            'model': 'vnptai_hackathon_small',
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            'max_completion_tokens': 50,
            'temperature': 0.0,
            'top_k': 1,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.small_headers, json=payload, timeout=120)
            self._check_rate_limit(response, "rerank_documents")
            return self._parse_response(response, "rerank_documents")
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[rerank_documents] Error: {e}")
            return ""

    # EMBEDDING model (vnpt_hackathon_embedding)
    def get_embedding(self, text: str) -> list:
        """Embedding text using vnptai_hackathon_embedding."""
        url = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
        payload = {
            'model': 'vnptai_hackathon_embedding',
            'input': text,
            'encoding_format': 'float',
        }
        try:
            response = requests.post(url, headers=self.embedding_headers, json=payload, timeout=60)
            self._check_rate_limit(response, "get_embedding")
            return response.json()['data'][0]['embedding']
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[get_embedding] Error: {e}")
            return [0.0] * 1024

client = VNPTClient()