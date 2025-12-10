import sys
import os
import json
import pandas as pd
from datetime import datetime
from src.config import config
from src.utils import load_data, transform_choices, save_transformed_data
from src.agent.graph import app
from src.client import RateLimitException

LOG_FILE = "inference_log.jsonl"
DETAIL_LOG_FILE = "inference_detail.log"
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

def append_detail_log(log_path: str, message: str):
    """Append detailed log message to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

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
        print(f"[ERROR] No input file found in {config.DATA_DIR}")
        return 1
    
    output_file = os.path.join(config.OUTPUT_DIR, "submission.csv")
    log_file = os.path.join(config.OUTPUT_DIR, LOG_FILE)
    detail_log = os.path.join(config.OUTPUT_DIR, DETAIL_LOG_FILE)
    emergency_csv = os.path.join(config.OUTPUT_DIR, EMERGENCY_CSV)
    
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    
    # Step 1: Load original data
    raw_data = load_data(input_file)
    
    # Step 2: Transform choices to "A. choice", "B. choice" format
    transformed_data = transform_choices(raw_data)
    
    # Step 3: Save transformed data
    save_transformed_data(transformed_data, input_file)
    
    # Load already processed questions for resume
    processed_qids = load_processed_qids(log_file)
    
    # Filter to only unprocessed questions
    remaining_data = [item for item in transformed_data if item.get('qid') not in processed_qids]
    
    total = len(transformed_data)
    done = len(processed_qids)
    remaining = len(remaining_data)
    
    print(f"Total: {total} | Done: {done} | Remaining: {remaining}")
    
    if remaining == 0:
        print("All questions already processed!")
        consolidate_log_to_csv(log_file, output_file)
        print(f"Submission saved to {output_file}")
        return 0
    
    print(f"Processing {remaining} questions...")
    append_detail_log(detail_log, f"=== Starting inference session ===")
    append_detail_log(detail_log, f"Total: {total}, Done: {done}, Remaining: {remaining}")
    
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
            
            # Detailed log to file
            append_detail_log(detail_log, f"{qid}: {final_state['answer']} ({final_state.get('category', '')})")
            
            # Simple console output
            current = done + idx + 1
            print(f"[{current}/{total}] {qid} -> {final_state['answer']}")
            
        except RateLimitException as e:
            # RATE LIMIT HANDLING
            print(f"\n{'='*50}")
            print(f"[RATE LIMIT] Quota exceeded at {qid}")
            print(f"{'='*50}")
            
            append_detail_log(detail_log, f"RATE LIMIT at {qid}: {e}")
            rate_limit_hit = True
            break
            
        except Exception as e:
            append_detail_log(detail_log, f"ERROR at {qid}: {e}")
            # Log with fallback answer
            record = {
                "qid": qid,
                "answer": "C",
                "category": "error",
                "reasoning": str(e)[:200]
            }
            append_to_log(log_file, record)
            current = done + idx + 1
            print(f"[{current}/{total}] {qid} -> C (error)")
    
    # CONSOLIDATE RESULTS
    if rate_limit_hit:
        count = consolidate_log_to_csv(log_file, emergency_csv)
        print(f"\nEmergency save: {count} answers -> {emergency_csv}")
        print(f"\n[HOW TO RESUME]")
        print(f"1. Wait for API quota reset")
        print(f"2. Run: uv run python predict.py")
        return 1
    else:
        count = consolidate_log_to_csv(log_file, output_file)
        print(f"\nDone! {count} answers -> {output_file}")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
