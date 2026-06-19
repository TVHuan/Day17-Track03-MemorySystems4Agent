from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    import json
    return json.loads(path.read_text(encoding='utf-8'))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for e in expected if e.lower() in ans_lower)
    if matches == len(expected):
        return 1.0
    if matches > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if "không biết" in answer.lower():
        return 0.0
    # dummy quality: just checking recall here for simplicity
    return recall_points(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_recall = 0.0
    total_quality = 0.0
    count = 0
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    max_memory_growth = 0
    
    for conv in conversations:
        user_id = conv.get("user_id", "default_user")
        thread_id = conv.get("id", conv.get("thread_id", "default_thread"))
        turns = conv.get("turns", conv.get("messages", []))
        
        for msg in turns:
            content = msg if isinstance(msg, str) else msg.get("content", "")
            agent.reply(user_id, thread_id, content)
            
        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        
        if hasattr(agent, 'compaction_count'):
            total_compactions += agent.compaction_count(thread_id)
        if hasattr(agent, 'memory_file_size'):
            max_memory_growth = max(max_memory_growth, agent.memory_file_size(user_id))
            
        if "recall_questions" in conv:
            for rq in conv["recall_questions"]:
                # New thread for recall
                recall_thread = thread_id + "_recall"
                reply = agent.reply(user_id, recall_thread, rq["question"])
                ans = reply["content"]
                
                expected = rq.get("expected_contains", rq.get("expected_facts", []))
                total_recall += recall_points(ans, expected)
                total_quality += heuristic_quality(ans, expected)
                count += 1
                
                total_agent_tokens += agent.token_usage(recall_thread)
                total_prompt_tokens += agent.prompt_token_usage(recall_thread)
    
    avg_recall = total_recall / count if count > 0 else 0.0
    avg_quality = total_quality / count if count > 0 else 0.0
    
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=max_memory_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    from tabulate import tabulate
    table_data = []
    headers = [
        "Agent", "Agent tokens only", "Prompt tokens processed", 
        "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"
    ]
    for r in rows:
        table_data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    # load datasets
    std_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")

    print("Running Standard Benchmark...")
    baseline_std = BaselineAgent(config, force_offline=True)
    advanced_std = AdvancedAgent(config, force_offline=True)
    
    row_b_std = run_agent_benchmark("Baseline", baseline_std, std_convs, config)
    row_a_std = run_agent_benchmark("Advanced", advanced_std, std_convs, config)
    print(format_rows([row_b_std, row_a_std]))
    
    print("\nRunning Long-Context Stress Benchmark...")
    baseline_stress = BaselineAgent(config, force_offline=True)
    advanced_stress = AdvancedAgent(config, force_offline=True)
    
    row_b_stress = run_agent_benchmark("Baseline", baseline_stress, stress_convs, config)
    row_a_stress = run_agent_benchmark("Advanced", advanced_stress, stress_convs, config)
    print(format_rows([row_b_stress, row_a_stress]))


if __name__ == "__main__":
    main()
