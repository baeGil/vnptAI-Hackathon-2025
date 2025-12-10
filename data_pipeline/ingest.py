from src.config import config
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from src.client import client

def ingest_data():
    print("Ingesting data...")
    # 1. Load Data (Wikipedia dumps, etc.)
    # 2. Split Text
    # 3. Embed Text using client.get_embedding()
    # 4. Save to Vector DB (FAISS/Chroma) at config.VECTOR_DB_PATH
    print(f"Data ingested to {config.VECTOR_DB_PATH}")

if __name__ == "__main__":
    ingest_data()
