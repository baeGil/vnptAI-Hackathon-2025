import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Embedding model credentials
    VNPT_EMBEDDING_API_KEY = os.getenv("VNPT_EMBEDDING_API_KEY")
    VNPT_EMBEDDING_TOKEN_KEY = os.getenv("VNPT_EMBEDDING_TOKEN_KEY")
    VNPT_EMBEDDING_TOKEN_ID = os.getenv("VNPT_EMBEDDING_TOKEN_ID")
    
    # LLM Large model credentials
    VNPT_LARGE_API_KEY = os.getenv("VNPT_LARGE_API_KEY")
    VNPT_LARGE_TOKEN_KEY = os.getenv("VNPT_LARGE_TOKEN_KEY")
    VNPT_LARGE_TOKEN_ID = os.getenv("VNPT_LARGE_TOKEN_ID")
    
    # LLM Small model credentials  
    VNPT_SMALL_API_KEY = os.getenv("VNPT_SMALL_API_KEY")
    VNPT_SMALL_TOKEN_KEY = os.getenv("VNPT_SMALL_TOKEN_KEY")
    VNPT_SMALL_TOKEN_ID = os.getenv("VNPT_SMALL_TOKEN_ID")
    
    # API Base URL
    VNPT_API_BASE_URL = os.getenv("VNPT_API_BASE_URL", "https://api.idg.vnpt.vn/data-service/v1/chat/completions")

    # Paths
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
    VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db")

config = Config()

# RAG Configuration
RAG_TOP_K = 10 # Number of documents to retrieve
RAG_RERANK_TOP_K = 3 # Number of documents to rerank
RAG_MIN_SCORE = 0.3 # Minimum score for documents to be considered (raise higher to filter out low quality documents)
RAG_MAX_CONTEXT_CHARS = 4000 # Maximum number of characters to include in context
RAG_MIN_CONTEXT_THRESHOLD = 500 # Minimum number of characters to include in context

# Math Configuration
MATH_MAX_RETRIES = 4 # Number of retries for math questions

# Router Priority Order
ROUTER_PRIORITY = ["toxic", "reading", "math", "rag"]

# Fast-track keywords
READING_KEYWORDS = [
    "đoạn thông tin", "đoạn văn", "bài đọc", "căn cứ vào đoạn",
    "theo đoạn", "cho đoạn văn", "cho đoạn thông tin", "dựa vào đoạn"
]

MATH_KEYWORDS = [
    "$", "\\frac", "^", "=", "tính giá trị", "biểu thức", "phương trình",
    "hàm số", "đạo hàm", "xác suất", "lãi suất", "vận tốc", "gia tốc",
    "điện trở", "gam", "mol", "nguyên tử khối", "gdp", "lạm phát",
    "tính tổng", "giải hệ", "tính diện tích", "tính thể tích", "bao nhiêu"
]

# Toxic/Refusal patterns
TOXIC_KEYWORDS = [
    "tôi không thể", "không thể cung cấp", "không thể trả lời",
    "nằm ngoài phạm vi", "trái pháp luật", "không hỗ trợ",
    "ngoài tầm hiểu biết", "bất hợp pháp", "không được phép"
]