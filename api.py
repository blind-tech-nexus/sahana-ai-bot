import json
import base64
import logging
import asyncio
import time
from typing import Optional
from config import CONTEXT_SIZE, MODEL_SAHANA_1, MODEL_SAHANA_2, MODEL_SAHANA_3, DEFAULT_MODEL
from api_keys import (
    fetch_api_keys, get_next_key_index, _gemini_request, HTTPException,
    normalize_mime_type, upload_file_to_gemini, upload_file_with_retry, delete_gemini_file,
    is_key_on_cooldown, mark_key_error, clear_key_error,
    compute_backoff_delay, get_rate_limiter,
)
from database import get_recent_history, save_message, get_user_temp, save_memory, get_memories, get_user_model
from markdown_parse import markdown_to_html, escape_html
from message import send_message, send_chat_action

logger = logging.getLogger("mero.api")
MAX_OUTPUT_TOKENS = 64000
MAX_INLINE_FILE_BYTES = 2 * 1024 * 1024  # 2MB threshold for Files API

FUNCTION_DECLARATIONS = [
    {
        "name": "save_memory",
        "description": "Save an important piece of information, fact, preference, or detail about the user to long-term memory for future reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The user ID of the current user."
                },
                "memory": {
                    "type": "string",
                    "description": "The important information to save about the user."
                }
            },
            "required": ["user_id", "memory"]
        }
    },
    {
        "name": "load_memory",
        "description": "Load and retrieve saved long-term memories for the user when you need context, facts, or recall about the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The user ID of the user whose memories to load."
                }
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "create_pdf",
        "description": "Create a downloadable PDF document on a given topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or subject content for the PDF document."
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "generate_image",
        "description": "Generate an AI image based on a text prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A detailed description of the image to generate."
                }
            },
            "required": ["prompt"]
        }
    }
]

async def get_gemini_model(chat_id: int) -> str:
    m = await get_user_model(chat_id)
    if m == "sahana-2":
        return MODEL_SAHANA_2
    if m == "sahana-3":
        return MODEL_SAHANA_3
    return MODEL_SAHANA_1

GEMINI_SUPPORTED_MIMES = {
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/gif",
    "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/opus", "audio/flac", "audio/aac", "audio/webm", "audio/m4a",
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska", "video/x-msvideo", "video/3gpp",
    "application/pdf", "text/plain", "text/html", "text/css", "text/javascript", "text/csv", "text/xml", "application/json", "text/markdown",
}

def normalize_mime_type(mime: str) -> str:
    mime = (mime or "").strip().lower()
    if mime in GEMINI_SUPPORTED_MIMES: return mime
    if mime.startswith("text/") or "javascript" in mime or "json" in mime or "xml" in mime: return "text/plain"
    return "text/plain"

def _normalize_part_keys(part: dict) -> dict:
    def _compact(data: dict) -> dict: return {k: v for k, v in data.items() if v not in ("", None)}
    if "inline_data" in part and isinstance(part["inline_data"], dict):
        ind = part["inline_data"]
        normalized = _compact({"mimeType": normalize_mime_type(ind.get("mime_type") or ind.get("mimeType")), "data": ind.get("data")})
        return {"inlineData": normalized} if normalized else {}
    if "inlineData" in part and isinstance(part["inlineData"], dict):
        ind = part["inlineData"]
        normalized = _compact({"mimeType": normalize_mime_type(ind.get("mimeType")), "data": ind.get("data")})
        return {"inlineData": normalized} if normalized else {}
    if "text" in part:
        text_val = (part.get("text") or "").strip()
        return {"text": text_val} if text_val else {}
    return part

def _normalize_parts(parts: list) -> list:
    normalized = []
    for part in parts:
        if isinstance(part, dict):
            candidate = _normalize_part_keys(part)
            normalized.append(candidate if candidate else part)
    return normalized


async def delete_gemini_file(file_uri: str) -> bool:
    """Delete a file from the Gemini Files API. Delegates to api_keys module."""
    from api_keys import delete_gemini_file as _delete
    return await _delete(file_uri)


def _friendly_error_message(raw_msg: str) -> str:
    """Convert raw API error JSON into user-friendly message."""
    if not raw_msg:
        return "All API keys are temporarily busy. Please try again in a moment."
    low = raw_msg.lower()
    if "429" in raw_msg or "resource_exhausted" in low or "quota" in low or "exceeded your current quota" in low:
        # Try to extract retry delay
        import re
        m = re.search(r'"retryDelay"\s*:\s*"([^"]+)"', raw_msg)
        if m:
            delay = m.group(1)
            return f"⏳ All API keys hit quota limits. Please retry in {delay}. (Automatic rotation exhausted all keys)"
        # Also check RetryInfo
        if "retry" in low:
            return "⏳ All API keys are rate-limited right now. Please wait ~30-60 seconds and try again. (Automatic key rotation tried all available keys)"
        return "⏳ Service is busy (quota exceeded). Please try again in ~30 seconds. All keys were rotated automatically."
    if "503" in raw_msg or "unavailable" in low or "overloaded" in low:
        return "⚠️ Gemini service is temporarily overloaded. Please try again in a few seconds."
    if "500" in raw_msg:
        return "⚠️ Gemini service error. Please try again shortly."
    if "401" in raw_msg or "403" in raw_msg or "permission" in low:
        return "⚠️ Some API keys are invalid or have permission issues. The system rotated to next keys but none succeeded."
    # Fallback: truncated raw but friendly prefix
    short = raw_msg[:500]
    # Avoid exposing full JSON with tons of details; give summary
    if len(short) > 200:
        short = short[:200] + "..."
    return short


async def try_api_call(model: str, body: dict) -> tuple[Optional[dict], Optional[str]]:
    """Execute a Gemini content generation request with key rotation.

    Uses full pool rotation (each key once). If all keys are on cooldown,
    waits for the soonest to recover and retries once. Returns (response_dict, error_message).
    """
    if not await fetch_api_keys():
        return None, "No API keys available"
    start_idx = await get_next_key_index()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data, err = await _gemini_request(url, body, start_idx)
    if data:
        return data, None
    raw_msg = err.message if isinstance(err, HTTPException) else (str(err) if err else "All keys exhausted")
    # If 429 and cooldown will clear soon, wait and retry one more full cycle
    low = raw_msg.lower() if isinstance(raw_msg, str) else ""
    if "429" in raw_msg or "resource_exhausted" in low or "quota" in low:
        try:
            from api_keys import time_until_next_key_available, get_available_keys
            wait = time_until_next_key_available()
            # If next key available within 12s, wait and retry once
            if 0 < wait <= 12.0:
                logger.info(f"All keys 429, waiting {wait:.1f}s for next key then retrying once")
                await asyncio.sleep(wait + 0.4)
                # Refresh start index
                start_idx2 = await get_next_key_index()
                data2, err2 = await _gemini_request(url, body, start_idx2)
                if data2:
                    return data2, None
                # Use second error if exists
                if err2:
                    raw_msg = err2.message if isinstance(err2, HTTPException) else str(err2)
        except Exception as exc:
            logger.debug(f"429 retry wait failed: {exc}")
    friendly = _friendly_error_message(raw_msg)
    return None, friendly

def build_body(history_messages: list[dict], current_parts: list, system_text: str, use_functions: bool = True) -> dict:
    raw_msgs = []
    for msg in history_messages:
        role = "user" if msg.get("role") == "user" else "model"
        text = (msg.get("text") or "").strip()
        if text: raw_msgs.append({"role": role, "parts": [{"text": text}]})
    
    normalized_current = _normalize_parts(current_parts)
    if normalized_current: raw_msgs.append({"role": "user", "parts": normalized_current})
    
    alternating_contents = []
    for msg in raw_msgs:
        role = msg["role"]
        parts = [p for p in msg["parts"] if p]
        if not parts: continue
        if alternating_contents and alternating_contents[-1]["role"] == role:
            alternating_contents[-1]["parts"].extend(parts)
        else:
            alternating_contents.append({"role": role, "parts": parts})
            
    if alternating_contents and alternating_contents[0]["role"] != "user": alternating_contents.pop(0)
    if not alternating_contents:
        alternating_contents.append({"role": "user", "parts": normalized_current or [{"text": "Hello"}]})
        
    body: dict = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": alternating_contents,
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 1.0},
    }
    # Only function calling is enabled — no built-in tools (googleSearch removed)
    if use_functions:
        body["tools"] = [{"functionDeclarations": FUNCTION_DECLARATIONS}]
    return body

def extract_sources(data: dict) -> list[dict]:
    # Built-in grounding removed — kept for backward compatibility
    return []

def extract_ai_text(data: dict) -> tuple[str, list[dict]]:
    candidates = data.get("candidates", [])
    if not candidates: return "No response received from AI.", []
    parts = candidates[0].get("content", {}).get("parts", [])
    ai_text = "\n".join(p["text"] for p in parts if p.get("text"))
    return (ai_text or "No response received from AI."), []

def extract_function_calls(data: dict) -> list[dict]:
    candidates = data.get("candidates", [])
    if not candidates: return []
    parts = candidates[0].get("content", {}).get("parts", [])
    calls = []
    for part in parts:
        fc = part.get("functionCall")
        if fc: calls.append({"name": fc.get("name", ""), "args": fc.get("args", {})})
    return calls

def format_response_with_sources(ai_text: str, sources: list[dict] = None) -> str:
    # Sources are no longer produced (googleSearch removed); keep function for compatibility
    return markdown_to_html(ai_text)

async def _execute_function(cid: int, func_name: str, args: dict) -> dict:
    if func_name == "save_memory":
        memory_text = args.get("memory", "")
        uid = args.get("user_id", cid)
        if memory_text:
            await save_memory(int(uid), memory_text)
            return {"status": "success", "message": f"Memory saved: {memory_text}"}
    elif func_name == "load_memory":
        uid = args.get("user_id", cid)
        memories = await get_memories(int(uid))
        if memories:
            return {"status": "success", "user_id": uid, "memories": memories}
        return {"status": "success", "user_id": uid, "memories": [], "message": "No saved memories found for this user."}
    elif func_name == "create_pdf":
        topic = args.get("topic", "")
        if topic:
            from texttopdf import execute_text_to_pdf
            await execute_text_to_pdf(cid, topic)
            return {"status": "success", "message": f"PDF created for topic: {topic}"}
    elif func_name == "generate_image":
        prompt = args.get("prompt", "")
        if prompt:
            from image_generation import execute_image
            await execute_image(cid, prompt, "User")
            return {"status": "success", "message": f"Image generated for: {prompt}"}
    return {"status": "error", "message": f"Unknown function or missing args: {func_name}"}

async def _send_function_response(cid: int, model: str, body: dict, function_calls: list[dict]) -> Optional[str]:
    contents = list(body.get("contents", []))
    fc_parts = [{"functionCall": {"name": fc["name"], "args": fc["args"]}} for fc in function_calls]
    contents.append({"role": "model", "parts": fc_parts})
    
    fr_parts = []
    for fc in function_calls:
        result = await _execute_function(cid, fc["name"], fc["args"])
        fr_parts.append({"functionResponse": {"name": fc["name"], "response": result}})
    contents.append({"role": "user", "parts": fr_parts})
    
    follow_up = dict(body)
    follow_up["contents"] = contents
    data, err = await try_api_call(model, follow_up)
    if not data: return None
    
    more_calls = extract_function_calls(data)
    if more_calls: return await _send_function_response(cid, model, follow_up, more_calls)
    
    ai_text, _ = extract_ai_text(data)
    return ai_text

async def call_gemini_raw(cid: int, parts: list, system_text: str) -> Optional[str]:
    if not await fetch_api_keys():
        return None
    model = await get_gemini_model(cid)

    processed_parts = await _process_parts_for_api(parts)

    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": _normalize_parts(processed_parts)}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.4},
    }
    data, err = await try_api_call(model, body)
    if not data:
        return None
    text, _ = extract_ai_text(data)
    return text


async def _process_parts_for_api(parts: list) -> list:
    """Convert large inline data parts to File API references.

    For files larger than MAX_INLINE_FILE_BYTES (2MB), upload to the Files API
    and replace inline data with a file URI reference.
    """
    processed = []
    for part in parts:
        if not isinstance(part, dict):
            processed.append(part)
            continue

        inline_data = part.get("inlineData") or part.get("inline_data")
        if inline_data and isinstance(inline_data, dict):
            file_data = inline_data.get("data")
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "application/octet-stream"

            if file_data and isinstance(file_data, str):
                try:
                    import base64 as _b64
                    decoded = _b64.b64decode(file_data)
                    file_size = len(decoded)

                    display_name = part.get("display_name") or f"file_{int(time.time())}"

                    if file_size > MAX_INLINE_FILE_BYTES:
                        file_uri = await upload_file_with_retry(decoded, mime_type, display_name)
                        if file_uri:
                            processed.append({"fileUri": file_uri, "mimeType": mime_type})
                            continue

                except Exception as exc:
                    logger.warning(f"File size check failed, falling back to inline: {exc}")

            processed.append(part)
            continue

        processed.append(part)
    return processed



async def handle_gemini(cid: int, current_parts: list, system_text: str, use_functions: bool = True, user_name: str = "User", **kwargs) -> Optional[str]:
    model = await get_gemini_model(cid)
    history = await get_recent_history(cid, CONTEXT_SIZE)

    processed_parts = await _process_parts_for_api(current_parts)

    # Only functionDeclarations are sent — all built-in tools (googleSearch) removed
    body = build_body(history, processed_parts, system_text, use_functions=use_functions)
    body["generationConfig"]["temperature"] = await get_user_temp(cid)
    
    if not await fetch_api_keys():
        msg = "Could not fetch API keys. Please try again later."
        await save_message(cid, "model", msg)
        await send_message(cid, msg)
        return None
        
    data, err = await try_api_call(model, body)
    if data:
        function_calls = extract_function_calls(data)
        if function_calls:
            for fc in function_calls:
                if fc["name"] in ("save_memory", "load_memory"): await send_chat_action(cid, "typing")
                elif fc["name"] == "create_pdf": await send_chat_action(cid, "upload_document")
                elif fc["name"] == "generate_image": await send_chat_action(cid, "upload_photo")
                
            final_text = await _send_function_response(cid, model, body, function_calls)
            if final_text:
                await save_message(cid, "model", final_text)
                if final_text not in ("No response received from AI.", "Failed to parse AI response."):
                    await send_message(cid, format_response_with_sources(final_text, []), parse_mode="HTML")
                else:
                    await send_message(cid, final_text)
                return final_text
                
        ai_text, sources = extract_ai_text(data)
        await save_message(cid, "model", ai_text)
        if ai_text not in ("No response received from AI.", "Failed to parse AI response."):
            await send_message(cid, format_response_with_sources(ai_text, sources), parse_mode="HTML")
        else:
            await send_message(cid, ai_text)
        return ai_text
        
    # err is already user-friendly from try_api_call
    friendly_err = err or "Unknown error occurred"
    # Log diagnostics
    try:
        from api_keys import get_cooldown_stats
        stats = get_cooldown_stats()
        logger.warning(f"handle_gemini failed cid={cid} err={friendly_err[:200]} stats={stats}")
    except Exception:
        pass
    # Don't prefix with "Error:" if message already has emoji/friendly prefix
    if friendly_err.strip().startswith(("⏳", "⚠️", "❌")):
        error_msg = friendly_err
    else:
        error_msg = f"❌ {friendly_err}"
    await save_message(cid, "model", error_msg)
    await send_message(cid, error_msg)
    return None


# web_search removed — googleSearch built-in tool is disabled.
# Only functionDeclarations are used for API requests.