import json
import base64
import logging
import httpx
import urllib.parse
from typing import Optional

from config import CONTEXT_SIZE, MODEL_LITE, MODEL_SMART
from api_keys import fetch_api_keys, get_next_key_index, KeyRotator, is_retriable_error
from database import (
    get_recent_history, save_message, get_user_temp,
    save_memory, get_user_model,
)
from markdown_parse import markdown_to_html, escape_html
from message import send_message, send_photo, send_chat_action

logger = logging.getLogger("mero.api")

MAX_OUTPUT_TOKENS = 64000

# ---------------------------------------------------------------------------
# Function declarations for Gemini native function calling
# ---------------------------------------------------------------------------
FUNCTION_DECLARATIONS = [
    {
        "name": "save_memory",
        "description": (
            "Save an important piece of information about the user to long-term memory. "
            "Use this when the user shares personal details, preferences, goals, birthday, "
            "name, location, or anything they want remembered across conversations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "The information to save as a memory, written as a concise fact.",
                },
            },
            "required": ["memory"],
        },
    },
    {
        "name": "create_pdf",
        "description": (
            "Create a PDF document on a given topic. Use when the user asks to create, "
            "generate, or make a PDF, document, or report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or subject matter for the PDF document.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "generate_image",
        "description": (
            "Generate an AI image based on a text prompt. Use when the user asks to generate, "
            "create, draw, or make an image, picture, illustration, or artwork."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A detailed description of the image to generate.",
                },
            },
            "required": ["prompt"],
        },
    },
]


def get_gemini_model(chat_id: int) -> str:
    """Return the appropriate model based on user settings."""
    m = get_user_model(chat_id)
    if m == "nepo-smart":
        return MODEL_SMART
    return MODEL_LITE


GEMINI_SUPPORTED_MIMES = {
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/gif",
    "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/opus", "audio/flac", "audio/aac", "audio/webm", "audio/x-wav", "audio/m4a", "audio/x-m4a",
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska", "video/x-msvideo", "video/3gpp",
    "application/pdf",
    "text/plain", "text/html", "text/css", "text/javascript", "text/csv", "text/xml", "application/json", "application/xml", "text/markdown",
}


def normalize_mime_type(mime: str) -> str:
    mime = (mime or "").strip().lower()
    if mime in GEMINI_SUPPORTED_MIMES:
        return mime
    if mime.startswith("text/") or "javascript" in mime or "json" in mime or "xml" in mime:
        return "text/plain"
    return "text/plain"


def _normalize_part_keys(part: dict) -> dict:
    def _compact(data: dict) -> dict:
        return {k: v for k, v in data.items() if v not in ("", None)}

    if "file_data" in part and isinstance(part["file_data"], dict):
        fd = part["file_data"]
        normalized = _compact({
            "mimeType": normalize_mime_type(fd.get("mime_type") or fd.get("mimeType")),
            "fileUri": fd.get("file_uri") or fd.get("fileUri"),
        })
        return {"fileData": normalized} if normalized else {}
    if "fileData" in part and isinstance(part["fileData"], dict):
        fd = part["fileData"]
        normalized = _compact({
            "mimeType": normalize_mime_type(fd.get("mimeType")),
            "fileUri": fd.get("fileUri"),
        })
        return {"fileData": normalized} if normalized else {}
    if "inline_data" in part and isinstance(part["inline_data"], dict):
        ind = part["inline_data"]
        normalized = _compact({
            "mimeType": normalize_mime_type(ind.get("mime_type") or ind.get("mimeType")),
            "data": ind.get("data"),
        })
        return {"inlineData": normalized} if normalized else {}
    if "inlineData" in part and isinstance(part["inlineData"], dict):
        ind = part["inlineData"]
        normalized = _compact({
            "mimeType": normalize_mime_type(ind.get("mimeType")),
            "data": ind.get("data"),
        })
        return {"inlineData": normalized} if normalized else {}
    if "text" in part:
        text_val = (part.get("text") or "").strip()
        return {"text": text_val} if text_val else {}
    return part


def _normalize_parts(parts: list) -> list:
    normalized: list = []
    for part in parts:
        if isinstance(part, dict):
            candidate = _normalize_part_keys(part)
            if candidate:
                normalized.append(candidate)
        else:
            normalized.append(part)
    return normalized


# ---------------------------------------------------------------------------
# API call with KeyRotator — proper sequential key iteration
# ---------------------------------------------------------------------------
async def try_api_call(
    body_json: str,
    model: str,
) -> tuple[Optional[str], Optional[str]]:
    """Try calling Gemini API with proper key rotation using KeyRotator."""
    if not await fetch_api_keys():
        return None, "No API keys available"

    start_idx = await get_next_key_index()
    rotator = KeyRotator(start_idx)

    last_error = None
    async with httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100)
    ) as client:
        while True:
            key = rotator.get_next_key()
            if key is None:
                break

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={key}"
            )
            try:
                resp = await client.post(
                    url,
                    content=body_json,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    return resp.text, None

                logger.warning("API call failed with status %d: %s", resp.status_code, resp.text)
                last_error = f"Status {resp.status_code}: {resp.text}"
            except Exception as exc:
                logger.warning("API call exception: %s", exc)
                last_error = str(exc)

    return None, last_error or "All keys exhausted"


def build_body(
    history_messages: list[dict],
    current_parts: list,
    system_text: str,
    use_tools: bool = True,
    use_functions: bool = True,
) -> dict:
    # Combine history and current_parts into a single list of messages to alternate
    raw_msgs = []
    for msg in history_messages:
        role = "user" if msg.get("role") == "user" else "model"
        text = (msg.get("text") or "").strip()
        if text:
            raw_msgs.append({"role": role, "parts": [{"text": text}]})

    # Append current user parts
    normalized_current = _normalize_parts(current_parts)
    if normalized_current:
        raw_msgs.append({"role": "user", "parts": normalized_current})

    # Merge consecutive roles and filter out empty parts
    alternating_contents = []
    for msg in raw_msgs:
        role = msg["role"]
        parts = [p for p in msg["parts"] if p]  # filter out empty parts
        if not parts:
            continue

        if alternating_contents and alternating_contents[-1]["role"] == role:
            # Merge parts into the previous message
            alternating_contents[-1]["parts"].extend(parts)
        else:
            alternating_contents.append({"role": role, "parts": parts})

    # Ensure the first message is "user"
    if alternating_contents and alternating_contents[0]["role"] != "user":
        alternating_contents.pop(0)

    # If empty, make sure we have at least the current user parts
    if not alternating_contents:
        if normalized_current:
            alternating_contents.append({"role": "user", "parts": normalized_current})
        else:
            alternating_contents.append({"role": "user", "parts": [{"text": "Hello"}]})

    body: dict = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": alternating_contents,
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 1.0},
    }

    if use_tools or use_functions:
        tools: list[dict] = []
        if use_tools:
            tools.append({"googleSearchRetrieval": {}})
        if use_functions:
            tools.append({"functionDeclarations": FUNCTION_DECLARATIONS})
        body["tools"] = tools
    return body


def extract_sources(data: dict) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()
    try:
        for chunk in (
            data.get("candidates", [{}])[0]
            .get("groundingMetadata", {})
            .get("groundingChunks", [])
        ):
            web = chunk.get("web", {})
            uri, title = web.get("uri", ""), web.get("title", "Source")
            if uri and uri not in seen:
                seen.add(uri)
                sources.append({"title": title.strip(), "url": uri.strip()})
    except Exception:
        pass
    return sources


def extract_ai_text(content: str) -> tuple[str, list[dict]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return "Failed to parse AI response.", []
    candidates = data.get("candidates", [])
    if not candidates:
        return "No response received from AI.", []
    parts = candidates[0].get("content", {}).get("parts", [])
    ai_text = "\n".join(p["text"] for p in parts if p.get("text"))
    return (ai_text or "No response received from AI."), extract_sources(data)


def extract_function_calls(content: str) -> list[dict]:
    """Extract function calls from a Gemini response.

    Returns a list of dicts with keys: name, args.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    candidates = data.get("candidates", [])
    if not candidates:
        return []
    parts = candidates[0].get("content", {}).get("parts", [])
    calls = []
    for part in parts:
        fc = part.get("functionCall")
        if fc:
            calls.append({
                "name": fc.get("name", ""),
                "args": fc.get("args", {}),
            })
    return calls


def format_response_with_sources(ai_text: str, sources: list[dict]) -> str:
    html = markdown_to_html(ai_text)
    if sources:
        html += "\n\n📌 <b>Sources:</b>\n"
        html += "".join(
            f'• <a href="{escape_html(s["url"])}">{escape_html(s["title"])}</a>\n'
            for s in sources
        )
    return html


# ---------------------------------------------------------------------------
# Function execution — called when Gemini returns functionCall
# ---------------------------------------------------------------------------
async def _execute_function(
    cid: int,
    name: str,
    func_name: str,
    args: dict,
) -> dict:
    """Execute a function call locally and return the result."""
    if func_name == "save_memory":
        memory_text = args.get("memory", "")
        if memory_text:
            save_memory(cid, memory_text)
            return {"status": "success", "message": f"Memory saved: {memory_text}"}
        return {"status": "error", "message": "No memory text provided"}

    if func_name == "create_pdf":
        topic = args.get("topic", "")
        if topic:
            from texttopdf import execute_text_to_pdf
            await execute_text_to_pdf(cid, topic)
            return {"status": "success", "message": f"PDF created for topic: {topic}"}
        return {"status": "error", "message": "No topic provided"}

    if func_name == "generate_image":
        prompt = args.get("prompt", "")
        if prompt:
            from image_generation import execute_image
            await execute_image(cid, prompt, name)
            return {"status": "success", "message": f"Image generated for: {prompt}"}
        return {"status": "error", "message": "No prompt provided"}

    return {"status": "error", "message": f"Unknown function: {func_name}"}


async def _send_function_response(
    cid: int,
    name: str,
    model: str,
    body: dict,
    function_calls: list[dict],
) -> Optional[str]:
    """Execute function calls, send results back to Gemini, return final response text."""
    # Build updated contents with function call and response
    contents = list(body.get("contents", []))

    # Add the model's function call as a model message
    fc_parts = []
    for fc in function_calls:
        fc_part: dict = {"functionCall": {"name": fc["name"], "args": fc["args"]}}
        fc_parts.append(fc_part)
    contents.append({"role": "model", "parts": fc_parts})

    # Execute each function and build response parts
    fr_parts = []
    for fc in function_calls:
        result = await _execute_function(cid, name, fc["name"], fc["args"])
        fr_part: dict = {
            "functionResponse": {
                "name": fc["name"],
                "response": result,
            }
        }
        fr_parts.append(fr_part)
    contents.append({"role": "user", "parts": fr_parts})

    # Build new request body with updated contents
    follow_up = dict(body)
    follow_up["contents"] = contents
    follow_up_json = json.dumps(follow_up)

    content, err = await try_api_call(follow_up_json, model)
    if not content:
        return None

    # Check if model wants to call more functions (max 1 follow-up round)
    more_calls = extract_function_calls(content)
    if more_calls:
        return await _send_function_response(cid, name, model, follow_up, more_calls)

    ai_text, _ = extract_ai_text(content)
    return ai_text


# ---------------------------------------------------------------------------
# Raw call (used by tools like refiner, translator, pdf)
# ---------------------------------------------------------------------------
async def call_gemini_raw(
    cid: int,
    parts: list,
    system_text: str,
) -> Optional[str]:
    if not await fetch_api_keys():
        return None
    model = get_gemini_model(cid)
    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": _normalize_parts(parts)}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.4},
    }
    content, err = await try_api_call(
        json.dumps(body), model
    )
    if not content:
        return None
    text, _ = extract_ai_text(content)
    return text


# ---------------------------------------------------------------------------
# Main handler — chat with function calling support
# ---------------------------------------------------------------------------
async def handle_gemini(
    cid: int,
    current_parts: list,
    system_text: str,
    use_tools: bool = True,
    use_functions: bool = True,
    user_name: str = "User",
) -> Optional[str]:
    """Handle a Gemini request with full function calling support.

    Flow:
      1. Send request with function declarations
      2. If response contains functionCall → execute locally → send result back
      3. Return final text response to user
    """
    model = get_gemini_model(cid)
    history = get_recent_history(cid, CONTEXT_SIZE)
    body = build_body(history, current_parts, system_text, use_tools, use_functions)
    body["generationConfig"]["temperature"] = get_user_temp(cid)

    if not await fetch_api_keys():
        msg = "Could not fetch API keys. Please try again later."
        save_message(cid, "model", msg)
        await send_message(cid, msg)
        return None

    content, err = await try_api_call(
        json.dumps(body), model
    )

    if content:
        # Check for function calls first
        function_calls = extract_function_calls(content)
        if function_calls:
            # Notify user that we're processing
            for fc in function_calls:
                fn = fc["name"]
                if fn == "save_memory":
                    await send_chat_action(cid, "typing")
                elif fn == "create_pdf":
                    await send_chat_action(cid, "upload_document")
                elif fn == "generate_image":
                    await send_chat_action(cid, "upload_photo")

            # Execute functions and get final response
            final_text = await _send_function_response(
                cid, user_name, model, body, function_calls
            )
            if final_text:
                save_message(cid, "model", final_text)
                if final_text not in (
                    "No response received from AI.",
                    "Failed to parse AI response.",
                ):
                    formatted = format_response_with_sources(final_text, [])
                    await send_message(cid, formatted, parse_mode="HTML")
                else:
                    await send_message(cid, final_text)
                return final_text
            # If function response failed, try to extract text from original response
            ai_text, sources = extract_ai_text(content)
            if ai_text not in (
                "No response received from AI.",
                "Failed to parse AI response.",
            ):
                save_message(cid, "model", ai_text)
                formatted = format_response_with_sources(ai_text, sources)
                await send_message(cid, formatted, parse_mode="HTML")
                return ai_text

        # No function calls — normal text response
        ai_text, sources = extract_ai_text(content)
        save_message(cid, "model", ai_text)
        if ai_text not in (
            "No response received from AI.",
            "Failed to parse AI response.",
        ):
            formatted = format_response_with_sources(ai_text, sources)
            await send_message(cid, formatted, parse_mode="HTML")
        else:
            await send_message(cid, ai_text)
        return ai_text

    error = f"Error: {err or 'Unknown error occurred'}"
    save_message(cid, "model", error)
    await send_message(cid, error)
    return None
