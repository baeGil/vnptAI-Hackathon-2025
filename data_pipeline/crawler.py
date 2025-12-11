#!/usr/bin/env python3
"""
Crawler for Vietnamese Legal Documents from vbpl.vn
Optimized for high-priority docs: Hiến pháp, Bộ luật, Luật
"""
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

# Document type IDs on vbpl.vn
DOC_TYPES = {
    "hien_phap": 15,
    "bo_luat": 16,
    "luat": 17,
    "nghi_dinh": 20,
    "thong_tu": 22,
}

HIGH_PRIORITY_TYPES = ["hien_phap", "bo_luat", "luat"]

BASE_URL = "https://vbpl.vn"
OUTPUT_DIR = "data/data_source"
CHECKPOINT_FILE = "data/data_source/checkpoint.json"
REQUEST_DELAY = 2

def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
    })
    return session

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"crawled_ids": [], "last_type": None, "last_page": 0}

def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f)

def get_document_list(session, doc_type_id, page=1):
    """Get list of documents by type from vbpl.vn."""
    url = f"{BASE_URL}/TW/Pages/vanban.aspx"
    params = {
        "idLoaiVanBan": doc_type_id,
        "dvid": 13,  # Trung ương
        "Page": page
    }
    
    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        documents = []
        # Find all document links with ItemID
        for link in soup.find_all('a', href=re.compile(r'ItemID=\d+')):
            href = link.get('href', '')
            match = re.search(r'ItemID=(\d+)', href)
            if match:
                doc_id = match.group(1)
                title = link.get_text(strip=True)
                if title and len(title) > 5:  # Skip empty/short links
                    full_url = f"{BASE_URL}/TW/Pages/vbpq-toanvan.aspx?ItemID={doc_id}"
                    documents.append({
                        "id": doc_id,
                        "title": title,
                        "url": full_url
                    })
        
        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in documents:
            if doc["id"] not in seen:
                seen.add(doc["id"])
                unique_docs.append(doc)
        
        # Check for next page
        has_next = bool(soup.find('a', string=re.compile(r'[›»]|Next|Tiếp')))
        
        return unique_docs, has_next
        
    except Exception as e:
        print(f"[Error] get_document_list: {e}")
        return [], False

def get_document_content(session, doc_url, doc_id):
    """Get full content of a document."""
    try:
        response = session.get(doc_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get title - try multiple selectors
        title = ""
        for selector in ['div.title h1', 'h1', '.doc-title', '#TitleVB']:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text(strip=True)
                if title:
                    break
        
        # Get metadata from info table
        info = {}
        for row in soup.select('table tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).replace(':', '').strip()
                value = cells[1].get_text(strip=True)
                if key and value:
                    info[key] = value
        
        # Get main content - the legal document text
        content = ""
        
        # Try different content containers
        for selector in ['#toanvancontent', '#Content', '.content', '.fulltext', '#vanban']:
            elem = soup.select_one(selector)
            if elem:
                # Get text preserving structure
                content = elem.get_text(separator='\n', strip=True)
                if len(content) > 100:
                    break
        
        # Fallback: get body text excluding navigation
        if len(content) < 100:
            # Remove nav/header/footer
            for tag in soup(['nav', 'header', 'footer', 'script', 'style', '.menu', '.navigation']):
                tag.decompose()
            
            main = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main:
                content = main.get_text(separator='\n', strip=True)
            else:
                # Last resort: get all paragraph text
                paragraphs = soup.find_all('p')
                content = '\n'.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
        
        # Clean content
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        if len(content) < 50:
            print(f"    Warning: Short content ({len(content)} chars)")
        
        return {
            "id": doc_id,
            "title": title or f"Document {doc_id}",
            "url": doc_url,
            "info": info,
            "content": content,
            "content_length": len(content),
            "crawled_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[Error] get_document_content {doc_id}: {e}")
        return None

def save_document(doc, doc_type):
    os.makedirs(f"{OUTPUT_DIR}/{doc_type}", exist_ok=True)
    filepath = f"{OUTPUT_DIR}/{doc_type}/{doc['id']}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return filepath

def crawl_document_type(session, doc_type, doc_type_id, checkpoint, limit=None):
    """Crawl all documents of a specific type."""
    print(f"\n{'='*50}")
    print(f"Crawling: {doc_type} (ID: {doc_type_id})")
    print(f"{'='*50}")
    
    page = 1
    total_crawled = 0
    
    if checkpoint.get("last_type") == doc_type:
        page = checkpoint.get("last_page", 1)
        print(f"Resuming from page {page}")
    
    while True:
        print(f"\n[Page {page}]")
        documents, has_next = get_document_list(session, doc_type_id, page)
        
        if not documents:
            print("No documents found on this page")
            break
        
        print(f"Found {len(documents)} documents")
        
        for doc_info in documents:
            doc_id = doc_info["id"]
            title = doc_info["title"]
            
            # Skip English documents
            if "tiếng anh" in title.lower() or "english" in title.lower():
                print(f"  [Skip English] {title[:40]}...")
                continue
            
            if doc_id in checkpoint["crawled_ids"]:
                continue
            
            print(f"  {title[:60]}...")
            
            doc = get_document_content(session, doc_info["url"], doc_id)
            
            if doc and doc.get("content") and len(doc["content"]) > 50:
                filepath = save_document(doc, doc_type)
                print(f"    ✓ Saved ({doc['content_length']} chars)")
                total_crawled += 1
                
                checkpoint["crawled_ids"].append(doc_id)
                checkpoint["last_type"] = doc_type
                checkpoint["last_page"] = page
                save_checkpoint(checkpoint)
                
                if limit and total_crawled >= limit:
                    print(f"\nReached limit: {limit}")
                    return total_crawled
            else:
                print(f"    ✗ Failed or empty content")
            
            time.sleep(REQUEST_DELAY)
        
        if not has_next:
            print("No more pages")
            break
        
        page += 1
    
    print(f"\nTotal for {doc_type}: {total_crawled}")
    return total_crawled

def main(types=None, limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    session = get_session()
    checkpoint = load_checkpoint()
    
    types_to_crawl = types or HIGH_PRIORITY_TYPES
    total = 0
    
    print(f"Starting at {datetime.now()}")
    print(f"Types: {types_to_crawl}")
    print(f"Output: {OUTPUT_DIR}")
    
    for doc_type in types_to_crawl:
        if doc_type not in DOC_TYPES:
            print(f"Unknown: {doc_type}")
            continue
        
        count = crawl_document_type(session, doc_type, DOC_TYPES[doc_type], checkpoint, limit)
        total += count
        
        if limit and total >= limit:
            break
    
    print(f"\n{'='*50}")
    print(f"Done at {datetime.now()}")
    print(f"Total: {total}")
    print(f"{'='*50}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--types', nargs='+', default=HIGH_PRIORITY_TYPES)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    if args.test:
        main(limit=5)
    else:
        main(types=args.types, limit=args.limit)
