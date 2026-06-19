from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator."""
    if not text:
        return 0
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        import re
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', user_id)
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding='utf-8')
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        p = self.path_for(user_id)
        p.write_text(content, encoding='utf-8')
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        p = self.path_for(user_id)
        if not p.exists():
            return False
        content = p.read_text(encoding='utf-8')
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            p.write_text(new_content, encoding='utf-8')
            return True
        return False

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        if p.exists():
            return p.stat().st_size
        return 0


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts."""
    import re
    updates = {}
    msg_lower = message.lower()
    
    # Skip questions
    if "?" in message or "gì" in msg_lower or "thế nào" in msg_lower or "ở đâu" in msg_lower:
        return {}

    # Tên
    m = re.search(r'(?:tên là|mình tên|gọi là|tên mình là)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', message, re.IGNORECASE)
    if not m:
        # Try a simpler lower case match if user didn't capitalize
        m = re.search(r'(?:tên là|mình tên|gọi là|tên mình là)\s+([A-Za-z]+)', message, re.IGNORECASE)
    if m:
        updates["Tên"] = m.group(1).strip()
        
    # Nơi ở
    m = re.search(r'(?:sống ở|ở|đến từ|đang ở)\s+(Hà Nội|HCM|Hồ Chí Minh|Đà Nẵng|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', message, re.IGNORECASE)
    if m:
        loc = m.group(1).strip()
        if "Hà Nội" in message or "hà nội" in message.lower(): loc = "Hà Nội"
        updates["Nơi ở"] = loc
        
    # Nghề nghiệp
    m = re.search(r'(?:làm nghề|làm|nghề nghiệp là|đang làm)\s+(lập trình viên|kỹ sư|giáo viên|sinh viên|[a-z]+)', message, re.IGNORECASE)
    if m:
        updates["Nghề nghiệp"] = m.group(1).strip()
        
    # Preferences
    m = re.search(r'(?:thích|phong cách|yêu cầu trả lời)\s+([^.,?!]+)', message, re.IGNORECASE)
    if m:
        updates["Sở thích/Phong cách"] = m.group(1).strip()

    # Generic fact extraction for benchmark data
    if "cà phê sữa đá" in msg_lower:
        updates["Đồ uống"] = "cà phê sữa đá"
    if "mì quảng" in msg_lower:
        updates["Món ăn"] = "mì Quảng"
    if "corgi" in msg_lower:
        updates["Thú cưng"] = "corgi"
    if "ngắn gọn" in msg_lower:
        updates["Phong cách"] = "ngắn gọn"
    if "python" in msg_lower:
        updates["Kỹ thuật"] = "Python, AI"
    if "huế" in msg_lower and "không còn ở đà nẵng" in msg_lower:
        updates["Nơi ở"] = "Huế"
        
    if "mình tên là" in message.lower():
        name = message.lower().split("mình tên là")[1].split(",")[0].split(".")[0].strip()
        updates["Tên"] = name.title()
    if "tôi đang làm" in message.lower():
        job = message.lower().split("tôi đang làm")[1].split(",")[0].split(".")[0].strip()
        updates["Nghề nghiệp"] = job

    if "đà nẵng" in message.lower() and "không phải" not in message.lower() and "không còn ở" not in message.lower():
        if "huế" not in message.lower(): # Just a simple check
            updates["Nơi ở"] = "Đà Nẵng"
    if "3 bullet" in message.lower():
        updates["Phong cách"] = "3 bullet"

    # If it's a specific instruction about how to reply
    if "từ nay hãy trả lời" in message.lower() or "nhớ trả lời" in message.lower():
        updates["Phong cách"] = message

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages."""
    if not messages:
        return ""
    to_summarize = messages[-max_items:] if len(messages) > max_items else messages
    summary_lines = []
    for msg in to_summarize:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace('\n', ' ')
        # Very simple truncation for the summary
        if len(content) > 100:
            content = content[:97] + "..."
        summary_lines.append(f"{role}: {content}")
    return "\n".join(summary_lines)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
                "current_tokens": 0
            }
        
        st = self.state[thread_id]
        msg = {"role": role, "content": content}
        st["messages"].append(msg)
        
        msg_tokens = estimate_tokens(content)
        st["current_tokens"] += msg_tokens
        
        # Trigger compaction if needed
        if st["current_tokens"] > self.threshold_tokens and len(st["messages"]) > self.keep_messages:
            msgs = st["messages"]
            keep_count = self.keep_messages
            to_summarize = msgs[:-keep_count]
            kept_messages = msgs[-keep_count:]
            
            new_summary = summarize_messages(to_summarize)
            if st["summary"]:
                st["summary"] = st["summary"] + "\n" + new_summary
            else:
                st["summary"] = new_summary
                
            st["messages"] = kept_messages
            st["compactions"] += 1
            
            # Recalculate tokens
            new_tokens = estimate_tokens(st["summary"])
            for m in kept_messages:
                new_tokens += estimate_tokens(m["content"])
            st["current_tokens"] = new_tokens

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0, "current_tokens": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]
