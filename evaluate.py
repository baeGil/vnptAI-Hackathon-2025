#!/usr/bin/env python3
"""
Compare ground truth answers with predicted answers.
Usage: python evaluate.py <ground_truth.json> <predictions.jsonl>
"""
import json
import sys

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
TICK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"

def load_ground_truth(filepath: str) -> dict:
    """Load ground truth from JSON file with 'answer' field."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['qid']: item.get('answer', '') for item in data}

def load_predictions(filepath: str) -> dict:
    """Load predictions from JSONL file."""
    predictions = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                predictions[record['qid']] = record.get('answer', '')
            except:
                pass
    return predictions

def evaluate(gt_path: str, pred_path: str):
    """Compare ground truth with predictions."""
    ground_truth = load_ground_truth(gt_path)
    predictions = load_predictions(pred_path)
    
    correct = 0
    wrong = 0
    missing = 0
    
    print(f"\n{'='*50}")
    print(f"Comparing: {gt_path} vs {pred_path}")
    print(f"{'='*50}\n")
    
    # Sort by qid
    all_qids = sorted(ground_truth.keys())
    
    for qid in all_qids:
        gt_answer = ground_truth[qid]
        pred_answer = predictions.get(qid, None)
        
        if pred_answer is None:
            print(f"  {qid}: GT={gt_answer} | Pred=? (missing)")
            missing += 1
        elif gt_answer.upper() == pred_answer.upper():
            print(f"{TICK} {qid}: {gt_answer}")
            correct += 1
        else:
            print(f"{CROSS} {qid}: GT={gt_answer} | Pred={pred_answer}")
            wrong += 1
    
    total = correct + wrong + missing
    evaluated = correct + wrong
    accuracy = (correct / evaluated * 100) if evaluated > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total questions: {total}")
    print(f"Evaluated: {evaluated}")
    print(f"Missing: {missing}")
    print(f"{GREEN}Correct: {correct}{RESET}")
    print(f"{RED}Wrong: {wrong}{RESET}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}\n") 

# Thực hiện đánh giá:
gt_path = "./data/val.json"
pred_path = "./output/inference_log.jsonl"
evaluate(gt_path, pred_path)