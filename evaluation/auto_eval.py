"""
Automated Evaluation — Tests agent quality against labeled datasets.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

DATASETS_DIR = Path("evaluation/datasets")
RESULTS_PATH = Path("memory/data/eval_results.jsonl")

SAMPLE_DATASET = [
    {
        "id": "eval_001",
        "input": "What are your pricing plans?",
        "expected_intent": "sales",
        "expected_agent": "sales",
        "tags": ["pricing", "sales"],
    },
    {
        "id": "eval_002",
        "input": "I can't connect to the API, getting 401 errors",
        "expected_intent": "support",
        "expected_agent": "support",
        "tags": ["api", "auth", "support"],
    },
    {
        "id": "eval_003",
        "input": "Tell me about multi-agent AI architectures",
        "expected_intent": "research",
        "expected_agent": "research",
        "tags": ["research", "ai"],
    },
    {
        "id": "eval_004",
        "input": "I want to schedule a demo of the Enterprise plan",
        "expected_intent": "sales",
        "expected_agent": "sales",
        "tags": ["demo", "enterprise", "sales"],
    },
    {
        "id": "eval_005",
        "input": "How do I reset my password?",
        "expected_intent": "support",
        "expected_agent": "support",
        "tags": ["auth", "support"],
    },
]


def _save_result(result: dict):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def run_evaluation(
    dataset: Optional[List[Dict]] = None,
    user_id: str = "eval_runner",
) -> Dict:
    """
    Run automated evaluation against a dataset.

    Args:
        dataset: List of eval cases (uses SAMPLE_DATASET if None)
        user_id: User ID to use in test requests

    Returns:
        Evaluation summary with accuracy metrics
    """
    from agents.orchestrator import process_message

    cases = dataset or SAMPLE_DATASET
    results = []
    correct_intent = 0
    correct_agent = 0

    for case in cases:
        start = time.time()
        try:
            response = process_message(
                user_id=user_id,
                message=case["input"],
            )
            latency = round((time.time() - start) * 1000, 2)

            predicted_intent = response.get("intent", "unknown")
            predicted_agent = response.get("agent", "unknown")

            intent_match = predicted_intent == case.get("expected_intent")
            agent_match = predicted_agent == case.get("expected_agent")

            if intent_match:
                correct_intent += 1
            if agent_match:
                correct_agent += 1

            result = {
                "id": case["id"],
                "input": case["input"],
                "expected_intent": case.get("expected_intent"),
                "predicted_intent": predicted_intent,
                "intent_correct": intent_match,
                "expected_agent": case.get("expected_agent"),
                "predicted_agent": predicted_agent,
                "agent_correct": agent_match,
                "confidence": response.get("confidence"),
                "latency_ms": latency,
                "error": None,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            result = {
                "id": case["id"],
                "input": case["input"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        results.append(result)
        _save_result(result)

    total = len(cases)
    summary = {
        "total_cases": total,
        "intent_accuracy": round(correct_intent / total, 3) if total else 0,
        "agent_accuracy": round(correct_agent / total, 3) if total else 0,
        "results": results,
        "run_at": datetime.now().isoformat(),
    }

    return summary


def load_dataset(name: str) -> List[Dict]:
    """Load a named dataset from evaluation/datasets/."""
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)
