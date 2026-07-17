"""
ChatGPT-style UI for the adaptive router demo — Embedding + Bandit variant.

Uses the semantic embedding classifier (Qwen3-Embedding-0.6B + pgvector)
feeding a Thompson-sampling bandit (see bandit.py) via LiteLLM's
async_pre_routing_hook.
...
Run:  python gradio_chat_ui_embedding.py
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

MOCK = os.environ.get("ROUTER_MOCK") == "1"

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
LITELLM_BASE = "http://localhost:4000"
OLLAMA_BASE = "http://localhost:11434"
FASTAPI_BASE = "http://localhost:8001"

ROUTER_MODEL = "qwen-semantic-router"

THINKING_MODELS = {"qwen3:8b", "openbmb/minicpm5"}

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
# Backend
# --------------------------------------------------------------------------- #
async def route_model(message: str) -> Dict[str, str]:
    """Ask the LiteLLM adaptive router which logical model to use."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{LITELLM_BASE}/v1/chat/completions",
                headers={
                    "Authorization": "Bearer anything",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ROUTER_MODEL,
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": 1,
                },
            )
        routed = resp.headers.get("x-litellm-adaptive-router-model")
        info = "\n".join(
            f"{k}: {v}"
            for k, v in resp.headers.items()
            if k.lower().startswith("x-litellm")
        )
        if routed is None:
            print("[route_model debug] WARNING: header missing on response, defaulting to large-model")
            routed = "large-model"
        return {"routed": routed, "info": info}
    except Exception as e:
        print(f"[route_model debug] EXCEPTION for message={message!r}: {type(e).__name__}: {e}")
        return {"routed": "large-model", "info": ""}


async def generate_title(message: str) -> str:
    """Summarize the first user message into a short sidebar title."""
    fallback = (message.strip()[:40] or "New chat")
    if MOCK:
        words = message.strip().split()
        return " ".join(words[:5])[:40] or "New chat"
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
                            "message. Reply with ONLY the title — no quotes, no "
                            "punctuation, no preamble.",
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


# --- Mock mode ------------------------------------------------------------- #
_COMPLEX_KW = (
    "code", "function", "implement", "algorithm", "sort", "prove", "calculate",
    "design", "architecture", "optimize", "debug", "class", "api", "build",
    "complexity", "differential", "probability", "schema",
)


def _mock_route(message: str) -> str:
    text = message.lower()
    if len(message) > 120 or any(
        re.search(rf"\b{kw}\b", text) for kw in _COMPLEX_KW
    ):
        return "large-model"
    return "small-model"


def _resolve_forced(mode: str) -> str:
    if mode == "small":
        return "small-model"
    if mode == "large":
        return "large-model"
    return ""


async def _mock_response(
    message: str, mode: str = "auto"
) -> AsyncGenerator[Tuple[str, str, str, str, str], None]:
    routed = _resolve_forced(mode) or _mock_route(message)
    ollama_model = MODEL_MAP.get(routed, "qwen3:8b")

    if routed == "large-model":
        think = "The user is asking for a coding task, routing to the larger model."
        reply = (
            "Here's a clean implementation:\n\n"
            "```python\ndef binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n            return mid\n"
            "        elif arr[mid] < target:\n            lo = mid + 1\n"
            "        else:\n            hi = mid - 1\n"
            "    return -1\n```\n\n"
            "Runs in **O(log n)** time."
        )
    else:
        think = "Simple factual question — routing to the small, fast model."
        reply = "The capital of France is **Paris**."

    yield "model", "", "", routed, ollama_model

    acc = ""
    for i in range(0, len(think), 4):
        acc += think[i: i + 4]
        yield "thinking", acc, "", routed, ollama_model
        await asyncio.sleep(0.01)

    racc = ""
    for i in range(0, len(reply), 4):
        racc += reply[i: i + 4]
        yield "reply", acc, racc, routed, ollama_model
        await asyncio.sleep(0.012)

    yield "done", acc, racc, routed, ollama_model


async def generate_response(
    message: str, history: List[Dict[str, str]], mode: str = "auto"
) -> AsyncGenerator[Tuple[str, str, str, str, str], None]:
    """Stream response. Yields (event, thinking, reply, routed, display_model)."""
    if MOCK:
        async for item in _mock_response(message, mode):
            yield item
        return

    forced = _resolve_forced(mode)
    if forced:
        routed = forced
    else:
        routing = await route_model(message)
        routed = routing["routed"]

    display_model = MODEL_MAP.get(routed, "qwen3:8b")
    identity = MODEL_IDENTITY.get(display_model, "You are a helpful AI assistant.")
    full_history = [{"role": "system", "content": identity + SYSTEM_SUFFIX}] + history

    yield "model", "", "", routed, display_model

    thinking = ""
    reply = ""

    # ---- Small model: call FastAPI server (non-streaming) ------------------
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
            yield "reply", "", reply, routed, display_model
        except Exception as e:
            reply = f"\n\n*⚠️ QwenRouterAI error: {e}*"
            yield "reply", "", reply, routed, display_model
        yield "done", "", reply, routed, display_model
        return

    # ---- Large model: stream from Ollama -----------------------------------
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
                out.append(
                    {
                        "role": "assistant",
                        "content": m["thinking"],
                        "metadata": {"title": "💭 Thinking"},
                    }
                )
            out.append({"role": "assistant", "content": m["content"]})
    if live:
        if live_thinking.strip():
            out.append(
                {
                    "role": "assistant",
                    "content": live_thinking,
                    "metadata": {"title": "💭 Thinking"},
                }
            )
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