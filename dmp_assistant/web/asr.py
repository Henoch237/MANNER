"""Speech-to-text for the web composer: DashScope qwen3-asr-flash.

Same API key as the planner (DASHSCOPE_API_KEY) - no extra credential, same
model family. The browser records a 16 kHz mono WAV while the mic button is
held and POSTs it to /api/asr; this module only wraps the transcription call.
"""
import os

import dashscope
from dashscope import MultiModalConversation

MODEL = "qwen3-asr-flash"


def transcribe_wav(audio_path):
    """audio file -> text. Language is auto-detected (English and Chinese
    both work). Raises RuntimeError on an API error (the route turns that
    into a clean JSON error)."""
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    messages = [{"role": "user", "content": [{"audio": audio_path}]}]
    resp = MultiModalConversation.call(
        model=MODEL,
        messages=messages,
        result_format="message",
    )

    if resp.status_code != 200:
        raise RuntimeError(f"DashScope error {resp.status_code}: "
                           f"{resp.code} - {resp.message}")

    # response.output.choices[0].message.content is a list of content parts
    parts = resp.output.choices[0].message.content
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
