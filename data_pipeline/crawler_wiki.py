#!/usr/bin/env python3
"""
Firecrawl Wiki Scraper
Crawl data from urls stored in data/wiki/url.txt file
Need to set FIRECRAWL_API_KEY in .env file

Usage:
    uv run data_pipeline/crawler_wiki.py
"""

import os
import re
import time
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("FIRECRAWL_API_KEY")
URL_FILE_PATH = "project/data/wiki/url.txt"
OUTPUT_DIR = "project/data/wiki/raw"
CHECKPOINT_FILE = "project/data/wiki/checkpoint.txt"

if not API_KEY:
    # Try to check if user has it in the old way or just warn
    print("Warning: FIRECRAWL_API_KEY not found in environment variables.")

def normalize_path(path):
    # Helper to resolve paths relative to project root or current dir
    return os.path.abspath(path)

def load_urls(file_path):
    """Parses the URL file which has a python-list-like structure with comments."""
    urls = []
    if not os.path.exists(file_path):
        print(f"Error: URL file not found at {file_path}")
        return []
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Extract URL from quotes
            # Matches "http..." or 'http...'
            match = re.search(r'[\'"](https?://.*?)[\'"]', line)
            if match:
                urls.append(match.group(1))
    
    return urls

def load_checkpoint(checkpoint_path):
    """Loads the set of already processed URLs."""
    processed = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                processed.add(line.strip())
    return processed

def append_checkpoint(checkpoint_path, url):
    """Appends a URL to the checkpoint file."""
    with open(checkpoint_path, 'a', encoding='utf-8') as f:
        f.write(f"{url}\n")

def clean_markdown(md):
    if not md: return ""
    
    # Step 1: Remove content before table of contents
    md = re.sub(r'^(?:Content:|\|?\s*Tính năng).*?(?=\*\*|[A-ZÀ-Ỹ])', '', md, flags=re.MULTILINE)
    md = re.sub(r'^Bách khoa toàn thư mở Wikipedia.*$', '', md, flags=re.MULTILINE)
    md = re.sub(r'^\|\s*[-]+\s*\|\s*[-]+\s*\|.*$', '', md, flags=re.MULTILINE)

    # Step 2: Remove links
    md = re.sub(r'\[\\\[\d+.*?\\\]\]\([^\)]*\)', '', md)
    md = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
    md = re.sub(r'\[([^\]]+)\]\([^\)]*redlink=[^\)]*\)', r'\1', md)

    # Step 3: Remove navigation box
    md = re.sub(r'\\?\[\s*sửa.*?\]\(.*?\)', '', md)
    md = re.sub(r'\\\[\s*\\\|', '', md)
    md = re.sub(r'\]\(\s*\|\s*', '', md)
    md = re.sub(r'\|\s*[-]+\s*\|\s*\n\|\s*_(Những sự kiện chính|Cải cách thể chế|Lịch sử hành chính).*', '', md, flags=re.MULTILINE | re.DOTALL)
    md = re.sub(r'^\s*Xem thêm:.*$', '', md, flags=re.MULTILINE)

    # Step 4: Clean formatting
    md = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', md)
    md = re.sub(r'<[^>]+>', '', md)
    md = re.sub(r'^\s*\|\s*', '', md, flags=re.MULTILINE)
    md = re.sub(r'\n{3,}', '\n\n', md)
    md = re.sub(r' +', ' ', md)

    return md.strip()

def post_process_cleanup(text):
    if not text: return ""

    # Step 1: Remove header/banner
    text = re.sub(r'^(Biểu quyết|Bài viết|Danh sách) (nội dung|chọn lọc|tốt).*?(\||”"\))\s*\|', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'^\|\s*\**Bài viết này.*?(giải quyết|thảo luận).*?\|\s*$', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'^\|\s*Bài viết hoặc đoạn này.*?\|\s*$', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'^Bước tới nội dung\s*', '', text)

    # Step 2: Remove navbox/side bar
    navbox_keywords = [
        r'^Loạt bài.*$', r'^Bản đồ Đông Nam Á.*$', r'^.*thời tiền sử"\).*$',
        r'^.*nền văn minh đầu tiên.*$', r'^.*Các vương quốc đầu tiên.*$',
        r'^.*Các quốc gia phong kiến.*$', r'^.*Giao lưu về văn hóa.*$',
        r'^.*Thực dân hóa.*$', r'^.*Xem thêm.*Lịch sử Brunei.*$'
    ]
    for pattern in navbox_keywords:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)

    # Step 3: Fix "sửa mã nguồn" and timeline number
    text = re.sub(r'\(\s*https?://[^\)]*\)', '', text)
    text = re.sub(r'\\\s*\[sửa.*?(?:\||sửa mã nguồn).*?\]', '', text)
    text = re.sub(r'\d{10,}.*Bản đồ.*$', '', text, flags=re.MULTILINE)

    # Step 4: Remove icon/file
    text = re.sub(r'^!Book icon.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*Tập tin:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*\.png.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*\.gif.*$', '', text, flags=re.MULTILINE)

    # Step 5: Fix quote error in table
    text = re.sub(r'"([^"]+)"\)', r'\1', text)
    text = re.sub(r'"\)\s*\|', ' |', text)

    # Step 6: Remove footer
    stop_headers = ["Tham khảo", "Xem thêm", "Liên kết ngoài", "Phụ lục"]
    for header in stop_headers:
        pattern = re.compile(r'^##\s+' + header, re.MULTILINE)
        match = pattern.search(text)
        if match:
            text = text[:match.start()]

    # Step 7: Format again 
    text = re.sub(r'^[:\-\| ]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def scrape_urls(app, urls, output_dir, checkpoint_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    processed_urls = load_checkpoint(checkpoint_path)
    print(f"Loaded {len(processed_urls)} URLs from checkpoint.")

    total = len(urls)
    print(f"Found {total} URLs in total.")

    # Configuration for Free Tier (10 req/min) => We need to ensure we don't exceed 1 request every 6 seconds.
    # Setting delay to 6.1 seconds to be safe.
    RATE_LIMIT_DELAY = 6.1 
    MAX_RETRIES = 5
    
    for i, url in enumerate(urls):
        if url in processed_urls:
            print(f"[{i+1}/{total}] Skipping (already scraped): {url}")
            continue

        print(f"\n[{i+1}/{total}] Scraping: {url}")
        
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                result = app.scrape(
                    url=url,
                    formats=["markdown"],
                    only_main_content=True,
                )
                
                if result and result.markdown and result.metadata and result.metadata.title:
                    cleaned_markdown = clean_markdown(result.markdown)
                    cleaned_markdown = post_process_cleanup(cleaned_markdown)
                    
                    title = result.metadata.title
                    # Sanitize filename
                    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
                    file_name = os.path.join(output_dir, f"{safe_title}.txt")
                    
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(cleaned_markdown)
                    print(f"Saved to {file_name}")
                    
                    # Update checkpoint
                    append_checkpoint(checkpoint_path, url) 
                    processed_urls.add(url)
                    break # Success, move to next URL
                    
                else:
                    print(f"Warning: No content or title found for {url}")
                    # Decide if we retry or skip. Let's skip for empty content.
                    break
                    
            except Exception as e:
                error_str = str(e)
                if "Rate Limit Exceeded" in error_str or "429" in error_str:
                    # Parse wait time if available, or default
                    # "retry after 7s"
                    wait_time = 15 # Default safe wait for backoff
                    match = re.search(r'retry after (\d+)s', error_str)
                    if match:
                        wait_time = int(match.group(1)) + 2 # Add buffer
                    
                    print(f"Rate Limit hit! Waiting {wait_time}s before retry ({attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    print(f"Error scraping {url}: {e}")
                    # Break on non-rate-limit errors to avoid infinite loops on 404s etc
                    break
    
        time.sleep(RATE_LIMIT_DELAY)

def main():
    if not API_KEY:
        print("Please set FIRECRAWL_API_KEY in .env file.")
        return

    app = FirecrawlApp(api_key=API_KEY)
    print("Firecrawl initialized!")
    
    # Resolve absolute path for data file
    if os.path.exists(URL_FILE_PATH):
        input_path = URL_FILE_PATH
        checkpoint_path = CHECKPOINT_FILE
    elif os.path.exists(os.path.join("..", URL_FILE_PATH)):
        input_path = os.path.join("..", URL_FILE_PATH)
        checkpoint_path = os.path.join("..", CHECKPOINT_FILE)
    else:
        # Fallback
        input_path = "/Users/AI/vnptAI/project/data/wiki/url.txt"
        checkpoint_path = "/Users/AI/vnptAI/project/data/wiki/checkpoint.txt"
    
    print(f"Reading URLs from: {input_path}")
    urls = load_urls(input_path)
    
    if not urls:
        print("No URLs found to processing.")
        return

    # Use absolute path for output to be safe
    output_path = os.path.abspath(OUTPUT_DIR)
    
    scrape_urls(app, urls, output_path, checkpoint_path)
    print("\nDone!")

if __name__ == "__main__":
    main()