from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            # Live path using LangGraph
            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke({"messages": [("user", message)]}, config)
            reply_content = result["messages"][-1].content
            
            # Approximate token counts for live mode to avoid failing benchmark signature
            out_tokens = estimate_tokens(reply_content)
            prompt_tokens = estimate_tokens(message)
            
            if thread_id not in self.sessions:
                self.sessions[thread_id] = SessionState()
            self.sessions[thread_id].token_usage += out_tokens
            self.sessions[thread_id].prompt_tokens_processed += prompt_tokens
            
            return {
                "content": reply_content,
                "token_usage": out_tokens,
                "prompt_tokens_processed": prompt_tokens
            }
        
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
            
        session = self.sessions[thread_id]
        
        # Calculate prompt context
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        prompt_tokens += estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens
        
        session.messages.append({"role": "user", "content": message})
        
        # Deterministic dummy reply for recall questions. 
        # Baseline doesn't have cross-thread memory.
        # So we just search in current session.messages for the answer.
        reply_content = "Tôi không biết."
        lower_msg = message.lower()
        if "tên" in lower_msg:
            # look for name in history
            for m in reversed(session.messages):
                if "tên là" in m["content"].lower() or "tên mình" in m["content"].lower():
                    reply_content = "Tôi nhớ rồi, " + m["content"]
                    break
        elif "sống" in lower_msg or "ở đâu" in lower_msg:
             for m in reversed(session.messages):
                if "sống ở" in m["content"].lower() or "đến từ" in m["content"].lower():
                    reply_content = "Bạn " + m["content"]
                    break
        elif "nghề" in lower_msg:
             for m in reversed(session.messages):
                if "làm nghề" in m["content"].lower() or "nghề nghiệp" in m["content"].lower():
                    reply_content = "Bạn " + m["content"]
                    break
        else:
            reply_content = "Ghi nhận thông tin của bạn."
            
        session.messages.append({"role": "assistant", "content": reply_content})
        
        # count output tokens
        out_tokens = estimate_tokens(reply_content)
        session.token_usage += out_tokens
        
        return {
            "content": reply_content,
            "token_usage": out_tokens,
            "prompt_tokens_processed": prompt_tokens
        }

    def _maybe_build_langchain_agent(self):
        try:
            # pyrefly: ignore [missing-import]
            from langgraph.prebuilt import create_react_agent
            # pyrefly: ignore [missing-import]
            from langgraph.checkpoint.memory import InMemorySaver
            
            llm = build_chat_model(self.config.model)
            memory = InMemorySaver()
            self.langchain_agent = create_react_agent(llm, tools=[], checkpointer=memory)
        except ImportError:
            pass
