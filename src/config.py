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
