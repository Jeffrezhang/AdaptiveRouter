"""
ChatGPT-style UI for the adaptive router demo — difficulty classifier version.

Calls classify_prompt() locally (no LiteLLM probe) to decide routing:
  - easy/medium -> QwenRouterAI (FastAPI on port 8001)
  - hard        -> qwen3:8b (Ollama streaming)

Run:  python gradio_chat_ui_difficulty.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Tuple

import gradio as gr
import httpx

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
OLLAMA_BASE = "http://localhost:11434"
FASTAPI_BASE = "http://localhost:8001"

THINKING_MODELS = {"qwen3:8b"}

MODEL_MAP = {
    "small-model": "QwenRouterAI",
    "large-model": "qwen3:8b",
}

MODEL_IDENTITY = {
    "QwenRouterAI": "You are QwenRouterAI, an AI assistant trained by China Unicom.",
    "qwen3:8b": "You are Qwen3, a large language model developed by Alibaba Cloud. You excel at complex reasoning, coding, and analytical tasks.",
}

SYSTEM_SUFFIX = (
    " Always respond in English only. Be concise and direct. Only answer the "
    "latest user message. Never repeat previous answers unless explicitly asked. "
    "For code requests, always provide complete working implementations."
)

WINDOW = 10

# --------------------------------------------------------------------------- #
# Local difficulty classifier — loaded once at startup
# --------------------------------------------------------------------------- #
_classify_fn = None

def _load_classifier():
    global _classify_fn
    try:
        from difficulty_classifier_inference import DifficultyClassifier
        clf = DifficultyClassifier()
        _classify_fn = clf.classify
        print("[gradio] Difficulty classifier loaded.")
    except Exception as e:
        print(f"[gradio] Failed to load difficulty classifier: {e}. Using fallback.")
        _classify_fn = None

_load_classifier()

# Category labeler for the difficulty classifier
_CATEGORY_RULES = [
    (re.compile(r"^\s*(implement|code|program|write)\b", re.IGNORECASE), "Coding"),
    (re.compile(r"\b(write|create|implement|build)\s+(?:a |an )?(?:python|javascript|java|sql|bash)\b", re.IGNORECASE), "Coding"),
    (re.compile(r"\b(write|create|implement|build)\b(?:\s+\w+){0,4}?\s+(function|class|method|script|api|endpoint)\b", re.IGNORECASE), "Coding"),
    (re.compile(r"\b(debug|fix|error|bug|exception|traceback)\b", re.IGNORECASE), "Coding"),
    (re.compile(r"\b(design|architect)\b.*\b(system|service|api|database|schema)\b", re.IGNORECASE), "Coding"),
    (re.compile(r"\b(solve|compute|calculate|prove)\b.*\b(equation|theorem|proof|problem)\b", re.IGNORECASE), "Closed QA"),
    (re.compile(r"\b(probability|statistics|combinatorics)\b", re.IGNORECASE), "Closed QA"),
    (re.compile(r"\b(write|draft|compose)\b.*\b(email|essay|blog|article|letter)\b", re.IGNORECASE), "Generation"),
    (re.compile(r"\b(brainstorm|ideas? for|suggest)\b", re.IGNORECASE), "Brainstorm"),
    (re.compile(r"\b(summarize|summary|tldr)\b", re.IGNORECASE), "Summarize"),
]

def _infer_category(text: str) -> str:
    for pattern, category in _CATEGORY_RULES:
        if pattern.search(text):
            return category
    return "Open QA"

def _route(message: str) -> str:
    """Returns 'small-model' or 'large-model'."""
    if _classify_fn is None:
        # Regex fallback
        hard = re.search(
            r"\b(dynamic programming|memoization|distributed system|raft|paxos|"
            r"concurren|cryptograph|neural network|backprop|gradient descent|"
            r"implement.{0,30}from scratch|system design)\b",
            message, re.IGNORECASE
        )
        return "large-model" if hard else "small-model"

    category = _infer_category(message)
    difficulty = _classify_fn(message, category)
    print(f"[classifier] category={category} difficulty={difficulty}")
    return "large-model" if difficulty == "hard" else "small-model"


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
async def generate_title(message: str) -> str:
    fallback = (message.strip()[:40] or "New chat")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{FASTAPI_BASE}/v1/chat/completions",
                json={
                    "model": "qwenrouterai",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Generate a 3-5 word title summarizing the user's "
                            "message. Reply with ONLY the title — no quotes, no preamble.",
                        },
                        {"role": "user", "content": message},
                    ],
                    "max_tokens": 24,
                },
            )
        title = (resp.json()["choices"][0]["message"].get("content", "") or "").strip()
        title = title.replace("\n", " ").strip().strip('"').strip("'").strip()
        return title[:40] or fallback
    except Exception:
        return fallback


async def generate_response(
    message: str, history: List[Dict[str, str]], mode: str = "auto"
) -> AsyncGenerator[Tuple[str, str, str, str, str], None]:
    """Yields (event, thinking, reply, routed, display_model)."""

    # Determine routing
    if mode == "small":
        routed = "small-model"
    elif mode == "large":
        routed = "large-model"
    else:
        routed = _route(message)

    display_model = MODEL_MAP.get(routed, "qwen3:8b")
    identity = MODEL_IDENTITY.get(display_model, "You are a helpful AI assistant.")
    full_history = [{"role": "system", "content": identity + SYSTEM_SUFFIX}] + history

    yield "model", "", "", routed, display_model

    thinking = ""
    reply = ""

    # ---- Small model: FastAPI (non-streaming) ------------------------------
    if routed == "small-model":
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
                resp = await client.post(
                    f"{FASTAPI_BASE}/v1/chat/completions",
                    json={
                        "model": "qwenrouterai",
                        "messages": full_history,
                        "max_tokens": 512,
                    },
                )
            reply = resp.json()["choices"][0]["message"].get("content", "") or ""
        except Exception as e:
            reply = f"\n\n*⚠️ QwenRouterAI error: {e}*"
        yield "reply", "", reply, routed, display_model
        yield "done", "", reply, routed, display_model
        return

    # ---- Large model: Ollama streaming ------------------------------------
    has_thinking = display_model in THINKING_MODELS
    in_thinking = False
    tag_buffer = ""

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": display_model,
                    "messages": full_history,
                    "stream": True,
                    "options": {"num_predict": 4096},
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue

                    msg = chunk.get("message", {})
                    reasoning = msg.get("thinking", "") or ""
                    if reasoning and has_thinking:
                        thinking += reasoning
                        yield "thinking", thinking, reply, routed, display_model
                        continue

                    content = msg.get("content", "") or ""
                    if not content:
                        continue

                    if not has_thinking:
                        reply += content
                        yield "reply", thinking, reply, routed, display_model
                        continue

                    tag_buffer += content
                    if "<think>" in tag_buffer:
                        in_thinking = True
                        tag_buffer = tag_buffer.replace("<think>", "")
                    if "</think>" in tag_buffer:
                        parts = tag_buffer.split("</think>", 1)
                        thinking += parts[0]
                        remaining = parts[1] if len(parts) > 1 else ""
                        tag_buffer = ""
                        in_thinking = False
                        if thinking.strip():
                            yield "thinking", thinking, reply, routed, display_model
                        if remaining:
                            reply += remaining
                            yield "reply", thinking, reply, routed, display_model
                        continue
                    if in_thinking:
                        thinking += tag_buffer
                        tag_buffer = ""
                        yield "thinking", thinking, reply, routed, display_model
                        continue
                    if tag_buffer:
                        reply += tag_buffer
                        tag_buffer = ""
                        yield "reply", thinking, reply, routed, display_model

        if tag_buffer and not in_thinking:
            reply += tag_buffer
    except Exception as e:
        reply += f"\n\n*⚠️ Ollama connection error: {e}*"
        yield "reply", thinking, reply, routed, display_model

    yield "done", thinking, reply, routed, display_model


# --------------------------------------------------------------------------- #
# View helpers
# --------------------------------------------------------------------------- #
def to_messages(
    conv_messages: List[Dict[str, str]],
    live_thinking: str = "",
    live_reply: str = "",
    live: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in conv_messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        else:
            if m.get("thinking", "").strip():
                out.append({
                    "role": "assistant",
                    "content": m["thinking"],
                    "metadata": {"title": "💭 Thinking"},
                })
            out.append({"role": "assistant", "content": m["content"]})
    if live:
        if live_thinking.strip():
            out.append({
                "role": "assistant",
                "content": live_thinking,
                "metadata": {"title": "💭 Thinking"},
            })
        out.append({"role": "assistant", "content": live_reply or " "})
    return out


def badge_md(routed: str = "", model: str = "") -> str:
    if not routed:
        return ""
    return f"<div class='badge'>routed&nbsp;→&nbsp;<b>{routed}</b>&nbsp;·&nbsp;{model}</div>"


def _uid() -> str:
    return uuid.uuid4().hex[:10]


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #
async def chat(message: str, chats: Dict[str, Any], current_id: str, mode: str = "auto"):
    chats = chats or {}
    if not message.strip():
        yield gr.update(), gr.skip(), gr.skip(), gr.update()
        return

    is_new = not current_id or current_id not in chats
    if is_new:
        current_id = _uid()
        chats[current_id] = {
            "title": "New chat",
            "created": datetime.now().timestamp(),
            "messages": [],
        }
    conv = chats[current_id]
    conv["messages"].append({"role": "user", "content": message})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in conv["messages"][-WINDOW:]
    ]

    yield (
        to_messages(conv["messages"]),
        chats,
        current_id,
        gr.update(value=""),
    )

    live_thinking = ""
    live_reply = ""
    routed = ""
    model = ""
    async for event, th, rp, rt, om in generate_response(message, history, mode):
        live_thinking, live_reply, routed, model = th, rp, rt, om
        yield (
            to_messages(conv["messages"], live_thinking, live_reply, live=True),
            gr.skip(),
            gr.skip(),
            gr.update(value=badge_md(routed, model)),
        )

    conv["messages"].append(
        {"role": "assistant", "content": live_reply, "thinking": live_thinking}
    )

    if is_new:
        conv["title"] = await generate_title(message)

    yield (
        to_messages(conv["messages"]),
        chats,
        current_id,
        gr.update(value=badge_md(routed, model)),
    )


def new_chat():
    return [], None, gr.update(value="")


def select_by_id(cid: str, chats: Dict[str, Any]):
    chats = chats or {}
    if not cid or cid not in chats:
        return [], None, gr.update(value="")
    conv = chats[cid]
    return to_messages(conv["messages"]), cid, gr.update(value="")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container {max-width: 100% !important; padding: 0 !important;}
#badge-row {min-height: 0;}
.badge {
    display:inline-block; font-size:12px; color:var(--body-text-color-subdued);
    border:1px solid var(--border-color-primary); border-radius:999px;
    padding:3px 12px; margin:4px 0;
}
.badge b {color:#10a37f;}
footer {display:none !important;}
#history-head {margin: 14px 0 2px; opacity:.6; font-size:12px; letter-spacing:.04em;}
.history-item {
    width:100% !important; justify-content:flex-start !important;
    background:transparent !important; border:1px solid transparent !important;
    box-shadow:none !important; border-radius:8px !important;
    padding:9px 12px !important; min-width:0 !important; min-height:0 !important;
    font-size:13.5px !important; font-weight:400 !important;
    color:var(--body-text-color) !important; transition:background .12s, border-color .12s;
}
.history-item span {
    overflow:hidden !important; text-overflow:ellipsis !important;
    white-space:nowrap !important; display:block !important;
    width:100% !important; text-align:left !important;
}
.history-item:hover {
    background:var(--background-fill-secondary) !important;
    border-color:var(--border-color-accent, #10a37f) !important;
}
.history-item.selected {background:var(--background-fill-secondary) !important;}
.history-item.selected span {font-weight:600 !important;}
"""

with gr.Blocks(title="Adaptive Router", fill_height=True) as demo:
    chats_state = gr.State({})
    current_id_state = gr.State(None)

    with gr.Column():
        badge = gr.Markdown("", elem_id="badge-row")
        chatbot = gr.Chatbot(
            height="72vh",
            show_label=False,
            avatar_images=(None, None),
            placeholder="<h2>How can I help?</h2><p>Simple questions route to the small model, complex ones to the large model.</p>",
            resizable=True,
        )
        msg = gr.Textbox(
            placeholder="Message Adaptive Router…  (Enter to send, Shift+Enter for newline)",
            show_label=False,
            submit_btn=True,
            lines=1,
            max_lines=8,
        )

    with gr.Sidebar(width=270):
        gr.Markdown("### ⚡ Adaptive Router")
        new_chat_btn = gr.Button("＋  New chat", variant="primary", size="sm")
        model_mode = gr.Radio(
            choices=[("Auto (router)", "auto"), ("Small", "small"), ("Large", "large")],
            value="auto",
            label="Model",
            elem_id="model-mode",
        )
        gr.Markdown("#### History", elem_id="history-head")

        @gr.render(inputs=[chats_state, current_id_state])
        def render_history(chats, current_id):
            items = sorted(
                (chats or {}).items(),
                key=lambda kv: kv[1]["created"],
                reverse=True,
            )
            if not items:
                gr.Markdown(
                    "<span style='opacity:.4; font-size:13px'>No conversations yet</span>"
                )
                return
            for cid, data in items:
                classes = ["history-item"]
                if cid == current_id:
                    classes.append("selected")
                item = gr.Button(
                    data.get("title") or "New chat",
                    size="sm",
                    elem_classes=classes,
                )
                item.click(
                    lambda chats, c=cid: select_by_id(c, chats),
                    inputs=[chats_state],
                    outputs=[chatbot, current_id_state, badge],
                )

    msg.submit(
        chat,
        [msg, chats_state, current_id_state, model_mode],
        [chatbot, chats_state, current_id_state, badge],
    ).then(lambda: "", outputs=[msg])
    new_chat_btn.click(
        new_chat, outputs=[chatbot, current_id_state, badge]
    )


if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
        css=CSS,
    )
