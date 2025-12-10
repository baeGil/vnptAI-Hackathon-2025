import sys
import os
import json
import pandas as pd
from src.config import config
from src.utils import load_data, transform_choices, save_transformed_data
from src.agent.graph import app
from src.client import RateLimitException

LOG_FILE = "inference_log.jsonl"
EMERGENCY_CSV = "submission_emergency.csv"

def load_processed_qids(log_path: str) -> set:
    """Load already processed question IDs from log file."""
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    processed.add(record.get('qid'))
                except:
                    pass
    return processed

def append_to_log(log_path: str, record: dict):
    """Append a processed record to the log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def consolidate_log_to_csv(log_path: str, csv_path: str):
    """Read all log entries and create a submission CSV."""
    results = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    results.append({
                        'qid': record.get('qid'),
                        'answer': record.get('answer', 'A')
                    })
                except:
                    pass
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"[Checkpoint] Consolidated {len(results)} results to {csv_path}")
    return len(results)

def main():
    # Determine input file path
    input_file = os.path.join(config.DATA_DIR, "val.json")
    
    # Fallbacks for testing
    if not os.path.exists(input_file):
        for fallback in ["public_test.json", "test.json", "private_test.json"]:
            fallback_path = os.path.join(config.DATA_DIR, fallback)
            if os.path.exists(fallback_path):
                input_file = fallback_path
                break
    
    if not os.path.exists(input_file):
        print(f"[Main] Error: No input file found in {config.DATA_DIR}")
        return 1
    
    output_file = os.path.join(config.OUTPUT_DIR, "submission.csv")
    log_file = os.path.join(config.OUTPUT_DIR, LOG_FILE)
    emergency_csv = os.path.join(config.OUTPUT_DIR, EMERGENCY_CSV)
    
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"[Main] Loading data from {input_file}...")
    
    # Step 1: Load original data
    raw_data = load_data(input_file)
    
    # Step 2: Transform choices to "A. choice", "B. choice" format
    print(f"[Main] Transforming {len(raw_data)} questions (mapping choices to A, B, C, D...)...")
    transformed_data = transform_choices(raw_data)
    
    # Step 3: Save transformed data
    save_transformed_data(transformed_data, input_file)
    
    # Load already processed questions for resume
    processed_qids = load_processed_qids(log_file)
    print(f"[Main] Found {len(processed_qids)} already processed questions.")
    
    # Filter to only unprocessed questions
    remaining_data = [item for item in transformed_data if item.get('qid') not in processed_qids]
    print(f"[Main] Processing {len(remaining_data)} remaining questions...")
    
    rate_limit_hit = False
    
    for idx, item in enumerate(remaining_data):
        qid = item.get("qid")
        question = item.get("question", "")
        choices = item.get("choices", [])
        
        # Initial State
        initial_state = {
            "question": question,
            "qid": qid,
            "choices": choices,
            "category": "",
            "context": "",
            "answer": "",
            "reasoning": ""
        }
        
        try:
            # Invoke Agent
            final_state = app.invoke(initial_state)
            
            record = {
                "qid": qid,
                "answer": final_state["answer"],
                "category": final_state.get("category", ""),
                "reasoning": final_state.get("reasoning", "")[:200]
            }
            
            # Save immediately to log
            append_to_log(log_file, record)
            
            print(f"[{idx+1}/{len(remaining_data)}] Processed {qid}: {final_state['answer']} ({final_state.get('category', '')})")
            
        except RateLimitException as e:
            # ============================================
            # RATE LIMIT HANDLING - Graceful Shutdown
            # ============================================
            print(f"\n{'='*60}")
            print(f"[RATE LIMIT] API quota exceeded at question {qid}")
            print(f"[RATE LIMIT] Error: {e}")
            print(f"{'='*60}")
            
            rate_limit_hit = True
            break
            
        except Exception as e:
            print(f"[Main] Error processing {qid}: {e}")
            # Log with fallback answer
            record = {
                "qid": qid,
                "answer": "C",  # Fallback
                "category": "error",
                "reasoning": str(e)[:200]
            }
            append_to_log(log_file, record)
    
    # ============================================
    # CONSOLIDATE RESULTS
    # ============================================
    if rate_limit_hit:
        print(f"\n[EMERGENCY] Rate limit detected. Creating emergency submission...")
        count = consolidate_log_to_csv(log_file, emergency_csv)
        print(f"[EMERGENCY] Saved {count} answers to {emergency_csv}")
        print(f"\n{'='*60}")
        print(f"[HOW TO RESUME]")
        print(f"1. Wait for API quota to reset (or switch tokens in .env)")
        print(f"2. Run: python predict.py")
        print(f"3. System will auto-detect {log_file}")
        print(f"   and continue from question {qid}")
        print(f"{'='*60}\n")
        return 1
    else:
        print(f"\n[SUCCESS] All {len(transformed_data)} questions processed!")
        consolidate_log_to_csv(log_file, output_file)
        print(f"[SUCCESS] Final submission saved to {output_file}")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
