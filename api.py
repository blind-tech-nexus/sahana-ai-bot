import json
import base64
import logging
import time
import random
import asyncio
import httpx
from typing import Optional
from config import CONTEXT_SIZE, MODEL_LITE, MODEL_SMART
from api_keys import (
    fetch_api_keys, get_next_key_index, KeyRotator,
    is_key_on_cooldown, mark_key_error, clear_key_error,
    is_retriable_error, get_retry_after, compute_backoff_delay,
    get_rate_limiter,
)
from database import get_recent_history, save_message, get_user_temp, save_memory, get_user_model, get_user_tools
from markdown_parse import markdown_to_html, escape_html
from message import send_message, send_chat_action

logger = logging.getLogger("mero.api")
MAX_OUTPUT_TOKENS = 64000
MAX_INLINE_FILE_BYTES = 2 * 1024 * 1024  # 2MB threshold for Files API

FUNCTION_DECLARATIONS = [
    {
        "name": "save_memory",
        "description": "Save an important piece of information about the user to long-term memory.",
        "parameters": {"type": "object", "properties": {"memory": {"type": "string", "description": "The information to save."}}, "required": ["memory"]}
    },
    {
        "name": "create_pdf",
        "description": "Create a PDF document on a given topic.",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "The topic for the PDF."}}, "required": ["topic"]}
    },
    {
        "name": "generate_image",
        "description": "Generate an AI image based on a text prompt.",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed description of the image."}}, "required": ["prompt"]}
    }
]

async def get_gemini_model(chat_id: int) -> str:
    m = await get_user_model(chat_id)
    if m in ("nepo-smart", "sahana-3"):
        return MODEL_SMART
    return MODEL_LITE

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


async def upload_file_to_gemini(
    file_bytes: bytes,
    mime_type: str,
    display_name: str = "file",
) -> Optional[str]:
    """Upload a file to the Gemini Files API and return the file URI.

    Uses the Files API for files larger than MAX_INLINE_FILE_BYTES (2MB).
    Files are automatically deleted after 48 hours.
    Returns the file URI (e.g., 'files/abc-123') or None on failure.
    """
    from api_keys import api_keys as _all_keys, get_next_key_index

    if not await fetch_api_keys():
        return None

    key_idx = await get_next_key_index()
    keys = _all_keys
    if not keys:
        return None

    mime_type = normalize_mime_type(mime_type)
    file_size = len(file_bytes)

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(min(3, len(keys))):
            key = keys[(key_idx + attempt) % len(keys)]
            upload_url = (
                "https://generativelanguage.googleapis.com/upload/v1beta/files"
                f"?key={key}&uploadType=resumable"
            )
            metadata = {
                "file": {
                    "display_name": display_name[:100],
                    "mime_type": mime_type,
                }
            }
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "X-Goog-Upload-File-Size": str(file_size),
                "X-Goog-Upload-Protocol": "resumable",
            }
            try:
                init_resp = await client.post(
                    upload_url,
                    content=json.dumps(metadata),
                    headers=headers,
                    timeout=30.0,
                )
                if init_resp.status_code != 200 and init_resp.status_code != 201:
                    if init_resp.status_code == 429:
                        mark_key_error(key)
                        delay = compute_backoff_delay(attempt, max_delay=15.0, jitter=True)
                        await asyncio.sleep(delay)
                        continue
                    logger.warning(
                        f"File upload init failed: {init_resp.status_code} {init_resp.text[:300]}"
                    )
                    continue

                upload_url_final = init_resp.headers.get("Location") or init_resp.headers.get("location")
                if not upload_url_final:
                    logger.warning("File upload: no upload URL returned")
                    continue

                upload_headers = {
                    "Content-Type": mime_type,
                    "X-Goog-Upload-File-Size": str(file_size),
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "upload, finalize",
                    "X-Goog-Upload-Offset": "0",
                }
                upload_resp = await client.post(
                    upload_url_final,
                    content=file_bytes,
                    headers=upload_headers,
                    timeout=120.0,
                )

                if upload_resp.status_code in (200, 201):
                    resp_data = upload_resp.json()
                    name = resp_data.get("name") or resp_data.get("file", {}).get("name")
                    if name:
                        return name
                    logger.warning(f"Upload returned no file name: {str(resp_data)[:300]}")
                else:
                    logger.warning(
                        f"File upload content failed: {upload_resp.status_code} {upload_resp.text[:300]}"
                    )
            except Exception as exc:
                logger.warning(f"File upload exception: {exc}")

    return None


async def upload_file_with_retry(
    file_bytes: bytes,
    mime_type: str,
    display_name: str = "file",
) -> Optional[str]:
    """Upload a file to Gemini Files API with retry logic.

    Returns the file URI or None on failure.
    """
    max_upload_retries = 3
    for attempt in range(max_upload_retries):
        result = await upload_file_to_gemini(file_bytes, mime_type, display_name)
        if result:
            return result
        if attempt < max_upload_retries - 1:
            delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
            await asyncio.sleep(delay)
    return None


async def delete_gemini_file(file_uri: str) -> bool:
    """Delete a file from the Gemini Files API.

    file_uri should be like 'files/abc-123'.
    """
    from api_keys import api_keys as _all_keys, get_next_key_index

    if not await fetch_api_keys():
        return False

    key = None
    idx = await get_next_key_index()
    keys = _all_keys
    if keys:
        key = keys[idx % len(keys)]

    if not key:
        return False

    url = f"https://generativelanguage.googleapis.com/v1beta/{file_uri}?key={key}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(url)
            return resp.status_code == 200
    except Exception as exc:
        logger.warning(f"Failed to delete Gemini file {file_uri}: {exc}")
        return False



async def try_api_call(body_json: str, model: str) -> tuple[Optional[str], Optional[str]]:
    if not await fetch_api_keys():
        return None, "No API keys available"
    start_idx = await get_next_key_index()
    from api_keys import api_keys as _all_keys
    rotator = KeyRotator(start_idx, _all_keys)
    last_error = None

    if not rotator.available_count:
        return None, "All API keys are on cooldown (rate limited)"

    limiter = get_rate_limiter()
    async with httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
    ) as client:
        attempt = 0
        max_retries = min(6, max(3, rotator.available_count))

        while attempt < max_retries:
            if not await limiter.acquire("gemini_api", timeout=30.0):
                last_error = "Rate limiter timeout waiting to acquire token"
                logger.warning(last_error)
                return None, last_error

            key = rotator.get_next_key()
            if key is None:
                if last_error is None:
                    last_error = "All API keys exhausted"
                break

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            try:
                resp = await client.post(
                    url,
                    content=body_json,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    clear_key_error(key)
                    return resp.text, None

                error_text = resp.text
                logger.warning(
                    f"API call failed with status {resp.status_code}: {error_text[:500]}"
                )

                if resp.status_code == 429:
                    retry_after = get_retry_after(resp)
                    mark_key_error(key)

                    if attempt < max_retries - 1:
                        if retry_after > 0:
                            base_delay = retry_after
                        else:
                            base_delay = compute_backoff_delay(attempt, max_delay=30.0, jitter=True)

                        logger.info(
                            f"429 on key index {start_idx + rotator.tried_count - 1}, "
                            f"retrying in {base_delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(base_delay)
                        attempt += 1
                        continue
                    last_error = f"Status 429: Rate limit exhausted after retries"
                    break

                if resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = compute_backoff_delay(attempt, max_delay=15.0, jitter=True)
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    last_error = f"Status {resp.status_code}: Server error"
                    break

                if resp.status_code in (400, 401, 403):
                    last_error = f"Status {resp.status_code}: {error_text}"
                    break

                last_error = f"Status {resp.status_code}: {error_text}"
                break

            except httpx.TimeoutException:
                last_error = "Request timed out"
                logger.warning(f"Timeout on API call (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning(f"API call exception: {exc}")
                if attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                break

    return None, last_error or "All keys exhausted"

def build_body(history_messages: list[dict], current_parts: list, system_text: str, use_tools: bool = True, use_functions: bool = True) -> dict:
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
    if use_tools or use_functions:
        tools: list[dict] = []
        if use_tools: tools.append({"googleSearch": {}})
        if use_functions: tools.append({"functionDeclarations": FUNCTION_DECLARATIONS})
        body["tools"] = tools
    return body

def extract_sources(data: dict) -> list[dict]:
    sources = []
    seen = set()
    try:
        for chunk in data.get("candidates", [{}])[0].get("groundingMetadata", {}).get("groundingChunks", []):
            web = chunk.get("web", {})
            uri, title = web.get("uri", ""), web.get("title", "Source")
            if uri and uri not in seen:
                seen.add(uri)
                sources.append({"title": title.strip(), "url": uri.strip()})
    except Exception: pass
    return sources

def extract_ai_text(content: str) -> tuple[str, list[dict]]:
    try: data = json.loads(content)
    except json.JSONDecodeError: return "Failed to parse AI response.", []
    candidates = data.get("candidates", [])
    if not candidates: return "No response received from AI.", []
    parts = candidates[0].get("content", {}).get("parts", [])
    ai_text = "\n".join(p["text"] for p in parts if p.get("text"))
    return (ai_text or "No response received from AI."), extract_sources(data)

def extract_function_calls(content: str) -> list[dict]:
    try: data = json.loads(content)
    except json.JSONDecodeError: return []
    candidates = data.get("candidates", [])
    if not candidates: return []
    parts = candidates[0].get("content", {}).get("parts", [])
    calls = []
    for part in parts:
        fc = part.get("functionCall")
        if fc: calls.append({"name": fc.get("name", ""), "args": fc.get("args", {})})
    return calls

def format_response_with_sources(ai_text: str, sources: list[dict]) -> str:
    html = markdown_to_html(ai_text)
    if sources:
        html += "\n\n📌 <b>Sources:</b>\n"
        html += "".join(f'• <a href="{escape_html(s["url"])}">{escape_html(s["title"])}</a>\n' for s in sources)
    return html

async def _execute_function(cid: int, func_name: str, args: dict) -> dict:
    if func_name == "save_memory":
        memory_text = args.get("memory", "")
        if memory_text:
            await save_memory(cid, memory_text)
            return {"status": "success", "message": f"Memory saved: {memory_text}"}
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
    content, err = await try_api_call(json.dumps(follow_up), model)
    if not content: return None
    
    more_calls = extract_function_calls(content)
    if more_calls: return await _send_function_response(cid, model, follow_up, more_calls)
    
    ai_text, _ = extract_ai_text(content)
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
    content, err = await try_api_call(json.dumps(body), model)
    if not content:
        return None
    text, _ = extract_ai_text(content)
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



async def handle_gemini(cid: int, current_parts: list, system_text: str, use_tools: bool = True, use_functions: bool = True, user_name: str = "User") -> Optional[str]:
    model = await get_gemini_model(cid)
    history = await get_recent_history(cid, CONTEXT_SIZE)

    user_tools = await get_user_tools(cid)
    web_search_enabled = user_tools.get("web_search", True) and use_tools

    processed_parts = await _process_parts_for_api(current_parts)

    body = build_body(history, processed_parts, system_text, use_tools=web_search_enabled, use_functions=use_functions)
    body["generationConfig"]["temperature"] = await get_user_temp(cid)
    
    if not await fetch_api_keys():
        msg = "Could not fetch API keys. Please try again later."
        await save_message(cid, "model", msg)
        await send_message(cid, msg)
        return None
        
    content, err = await try_api_call(json.dumps(body), model)
    if content:
        function_calls = extract_function_calls(content)
        if function_calls:
            for fc in function_calls:
                if fc["name"] == "save_memory": await send_chat_action(cid, "typing")
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
                
        ai_text, sources = extract_ai_text(content)
        await save_message(cid, "model", ai_text)
        if ai_text not in ("No response received from AI.", "Failed to parse AI response."):
            await send_message(cid, format_response_with_sources(ai_text, sources), parse_mode="HTML")
        else:
            await send_message(cid, ai_text)
        return ai_text
        
    error = f"Error: {err or 'Unknown error occurred'}"
    await save_message(cid, "model", error)
    await send_message(cid, error)
    return None


async def web_search(query: str, cid: int) -> dict:
    """Execute a standalone web search query using Gemini Google Search Grounding."""
    system_text = "Search the web and provide detailed, accurate results with sources."
    parts = [{"text": f"Search the web for: {query}"}]
    model = await get_gemini_model(cid)
    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": parts}],
        "tools": [{"googleSearch": {}}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.3},
    }
    content, err = await try_api_call(json.dumps(body), model)
    if not content:
        return {"status": "error", "message": err or "Failed to search web."}
    ai_text, sources = extract_ai_text(content)
    return {"status": "success", "results": ai_text, "sources": sources}