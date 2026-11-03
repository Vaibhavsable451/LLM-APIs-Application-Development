"""
AI Chat Application - Streamlit frontend + Groq API
Model: openai/gpt-oss-120b

Covers the module objectives:
1. API auth via environment variables + Python client
2. Chat history, system/user/assistant roles, model parameters, token usage
3. Structured (JSON) responses, error handling, cost tracking
4. Chat UI with custom instructions
"""
import json
import os
import time

import streamlit as st
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

# ---------- Config ----------
load_dotenv()  # loads GROQ_API_KEY from a local .env file
MODEL = "openai/gpt-oss-120b"

# Approximate Groq pricing in USD per 1M tokens (edit if pricing changes)
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise AI assistant."

st.set_page_config(page_title="AI Chat", page_icon="💬", layout="centered")


# ---------- Helpers ----------
def get_client() -> Groq | None:
    """Create the Groq client from env var or sidebar input."""
    key = os.getenv("GROQ_API_KEY") or st.session_state.get("manual_key")
    return Groq(api_key=key) if key else None


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_INPUT_PER_M + completion_tokens * PRICE_OUTPUT_PER_M) / 1_000_000


def build_messages(system_prompt: str, history: list[dict], json_mode: bool) -> list[dict]:
    """System message first, then the full user/assistant history."""
    sys = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    if json_mode:
        sys += (
            "\n\nRespond ONLY with a valid JSON object with keys: "
            '"answer" (string), "confidence" (low|medium|high), "follow_ups" (list of 2 strings).'
        )
    return [{"role": "system", "content": sys}] + history


def call_llm(client: Groq, messages, temperature, max_tokens, top_p, json_mode, retries=3):
    """Call the API with exponential backoff on rate limits / network errors."""
    kwargs = dict(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, ...


# ---------- Session state ----------
st.session_state.setdefault("messages", [])  # [{"role": "user"|"assistant", "content": str}]
st.session_state.setdefault("tokens_in", 0)
st.session_state.setdefault("tokens_out", 0)
st.session_state.setdefault("cost", 0.0)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    if not os.getenv("GROQ_API_KEY"):
        st.text_input("Groq API key", type="password", key="manual_key",
                      help="Or set GROQ_API_KEY in a .env file.")
    st.caption(f"Model: `{MODEL}`")

    system_prompt = st.text_area("Custom instructions (system prompt)",
                                 value=DEFAULT_SYSTEM_PROMPT, height=120)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Max output tokens", 64, 4096, 1024, 64)
    json_mode = st.toggle("Structured JSON response", value=False)

    st.divider()
    st.subheader("📊 Usage (this session)")
    st.metric("Input tokens", st.session_state.tokens_in)
    st.metric("Output tokens", st.session_state.tokens_out)
    st.metric("Est. cost (USD)", f"${st.session_state.cost:.5f}")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.update(messages=[], tokens_in=0, tokens_out=0, cost=0.0)
        st.rerun()

# ---------- Main UI ----------
st.title("💬 AI Chat Application")
st.caption("Powered by Groq · openai/gpt-oss-120b")

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle new input
if prompt := st.chat_input("Type your message..."):
    client = get_client()
    if client is None:
        st.error("No API key found. Add GROQ_API_KEY to .env or enter it in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                resp = call_llm(
                    client,
                    build_messages(system_prompt, st.session_state.messages, json_mode),
                    temperature, max_tokens, top_p, json_mode,
                )
            reply = resp.choices[0].message.content or ""

            # Structured output: parse + render nicely, fall back to raw text
            if json_mode:
                try:
                    data = json.loads(reply)
                    st.markdown(data.get("answer", ""))
                    st.caption(f"Confidence: **{data.get('confidence', 'n/a')}**")
                    for q in data.get("follow_ups", []):
                        st.markdown(f"- 💡 {q}")
                    with st.expander("Raw JSON"):
                        st.json(data)
                except json.JSONDecodeError:
                    st.warning("Model did not return valid JSON; showing raw text.")
                    st.markdown(reply)
            else:
                st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

            # Token + cost tracking
            u = resp.usage
            st.session_state.tokens_in += u.prompt_tokens
            st.session_state.tokens_out += u.completion_tokens
            st.session_state.cost += estimate_cost(u.prompt_tokens, u.completion_tokens)
            st.rerun()  # refresh sidebar metrics

        except AuthenticationError:
            st.error("🔑 Invalid API key. Check your GROQ_API_KEY.")
        except RateLimitError:
            st.error("⏳ Rate limit hit even after retries. Wait a moment and try again.")
        except APIConnectionError:
            st.error("🌐 Network problem reaching Groq. Check your connection.")
        except APIStatusError as e:
            st.error(f"API error {e.status_code}: {e.message}")
        except Exception as e:  # last-resort safety net
            st.error(f"Unexpected error: {e}")
