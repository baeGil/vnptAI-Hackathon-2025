import requests
from .config import config

class RateLimitException(Exception):
    """Exception raised when API rate limit (429) or quota exceeded (401) is hit."""
    pass

class VNPTClient:
    """
    VNPT AI API Client với các method riêng cho từng tác vụ.
    Tự động phát hiện Rate Limit (429/401) và raise RateLimitException.
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
            error_msg = f"[{method_name}] Rate Limit/Quota exceeded: HTTP {response.status_code}"
            print(error_msg)
            raise RateLimitException(error_msg)
        response.raise_for_status()

    # ROUTER CLASSIFICATION (vnpt_hackathon_small)
    def classify_router(self, prompt: str) -> str:
        """Phân loại câu hỏi sử dụng vnpt_hackathon_small."""
        url = f"{self.base_url}/vnptai-hackathon-small"
        payload = {
            'model': 'vnptai_hackathon_small',
            'messages': [{"role": "user", "content": prompt}],
            'max_completion_tokens': 20,
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 1,
            'n': 1,
            'seed': 42,
            'response_format': {'type': 'json_object'},
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.small_headers, json=payload, timeout=30)
            self._check_rate_limit(response, "classify_router")
            return response.json()['choices'][0]['message']['content']
        except RateLimitException:
            raise  # Re-raise to be caught by predict.py
        except Exception as e:
            print(f"[classify_router] Error: {e}")
            return ""

    # MATH CODE GENERATION (vnpt_hackathon_large)
    def generate_math_code(self, prompt: str) -> str:
        """Sinh code Python sử dụng vnpt_hackathon_large."""
        url = f"{self.base_url}/vnptai-hackathon-large"
        payload = {
            'model': 'vnptai_hackathon_large',
            'messages': [
                {"role": "system", "content": "Bạn là chuyên gia lập trình code Python. Luôn trả về code trong markdown code block."},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': 2048,
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 50,
            'frequency_penalty': 0.0,
            "presence_penalty": 0.0,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=60)
            self._check_rate_limit(response, "generate_math_code")
            return response.json()['choices'][0]['message']['content']
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[generate_math_code] Error: {e}")
            return ""

    # MATH ANSWER SELECTION (vnpt_hackathon_small)
    def select_math_answer(self, prompt: str) -> str:
        """Chọn đáp án cuối cùng cho bài toán sử dụng vnpt_hackathon_large."""
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
            'top_k': 40,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'n': 1,
            'seed': 100,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=30)
            self._check_rate_limit(response, "select_math_answer")
            return response.json()['choices'][0]['message']['content']
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[select_math_answer] Error: {e}")
            return ""

    # RAG GENERATION (vnpt_hackathon_large)
    def generate_rag_answer(self, prompt: str) -> str:
        """Sinh câu trả lời RAG sử dụng vnpt_hackathon_large."""
        url = f"{self.base_url}/vnptai-hackathon-large"
        payload = {
            'model': 'vnptai_hackathon_large',
            'messages': [
                {"role": "system", "content": "Bạn là trợ lý tri thức. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp và kiến thức của bạn."},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': 10,
            'temperature': 0.0,
            'top_p': 0.85,
            'top_k': 30,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=60)
            self._check_rate_limit(response, "generate_rag_answer")
            return response.json()['choices'][0]['message']['content']
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[generate_rag_answer] Error: {e}")
            return ""

    # READING COMPREHENSION (vnpt_hackathon_large)
    def generate_reading_answer(self, prompt: str) -> str:
        """Đọc hiểu và trả lời sử dụng vnpt_hackathon_large."""
        url = f"{self.base_url}/vnptai-hackathon-large"
        payload = {
            'model': 'vnptai_hackathon_large',
            'messages': [
                {"role": "system", "content": "Bạn là chuyên gia đọc hiểu văn bản. Hãy đọc kỹ đoạn văn và trả lời chính xác theo nội dung."},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': 10,
            'temperature': 0.0,
            'top_p': 0.85,
            'top_k': 30,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'n': 1,
            'seed': 42,
            'stream': False
        }
        try:
            response = requests.post(url, headers=self.large_headers, json=payload, timeout=60)
            self._check_rate_limit(response, "generate_reading_answer")
            return response.json()['choices'][0]['message']['content']
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[generate_reading_answer] Error: {e}")
            return ""

    # EMBEDDING (vnpt_hackathon_embedding)
    def get_embedding(self, text: str) -> list:
        """Embedding text sử dụng vnptai_hackathon_embedding."""
        url = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
        payload = {
            'model': 'vnptai_hackathon_embedding',
            'input': text,
            'encoding_format': 'float',
        }
        try:
            response = requests.post(url, headers=self.embedding_headers, json=payload, timeout=15)
            self._check_rate_limit(response, "get_embedding")
            return response.json()['data'][0]['embedding']
        except RateLimitException:
            raise
        except Exception as e:
            print(f"[get_embedding] Error: {e}")
            return [0.0] * 1024

client = VNPTClient()