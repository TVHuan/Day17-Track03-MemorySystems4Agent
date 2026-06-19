from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    from model_provider import ProviderConfig
    from config import LabConfig
    model = ProviderConfig("gemini", "gemini-1.5-flash", 0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=20, # small for tests
        compact_keep_messages=2,
        model=model,
        judge_model=model
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    from memory_store import UserProfileStore
    store = UserProfileStore(tmp_path / "profiles")
    
    # write
    store.write_text("u1", "- Tên: John\n- Nơi ở: Hanoi")
    assert store.file_size("u1") > 0
    
    # read
    txt = store.read_text("u1")
    assert "John" in txt
    
    # edit
    store.edit_text("u1", "Hanoi", "HCMC")
    txt = store.read_text("u1")
    assert "HCMC" in txt


def test_compact_trigger(tmp_path: Path) -> None:
    from memory_store import CompactMemoryManager
    cm = CompactMemoryManager(threshold_tokens=20, keep_messages=2)
    # A single message is ~len/4 tokens. A 80 char string is 20 tokens.
    cm.append("t1", "user", "A" * 100)
    cm.append("t1", "assistant", "B" * 100)
    cm.append("t1", "user", "C" * 100)
    
    assert cm.compaction_count("t1") > 0
    ctx = cm.context("t1")
    assert ctx["summary"] != ""
    assert len(ctx["messages"]) == 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    base = BaselineAgent(config, force_offline=True)
    adv = AdvancedAgent(config, force_offline=True)
    
    msg1 = "Xin chào, mình tên là Alice"
    base.reply("user1", "t1", msg1)
    adv.reply("user1", "t1", msg1)
    
    msg2 = "Mình tên gì?"
    base_ans = base.reply("user1", "t2", msg2)["content"]
    adv_ans = adv.reply("user1", "t2", msg2)["content"]
    
    # Baseline should not know across thread t1 -> t2
    assert "alice" not in base_ans.lower()
    # Advanced should know
    assert "alice" in adv_ans.lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    base = BaselineAgent(config, force_offline=True)
    adv = AdvancedAgent(config, force_offline=True)
    
    for i in range(10):
        msg = f"Turn {i} with some extra text so it takes some tokens. " * 5
        base.reply("user1", "t1", msg)
        adv.reply("user1", "t1", msg)
        
    p1 = base.prompt_token_usage("t1")
    p2 = adv.prompt_token_usage("t1")
    
    # Advanced uses summarization, so its prompt context processing load should grow much slower
    assert p2 < p1
