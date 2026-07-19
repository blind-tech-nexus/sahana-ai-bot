import json
import base64
import logging
import httpx
import asyncio
from typing import Optional
from config import CONTEXT_SIZE, MODEL_SAHANA_1, MODEL_SAHANA_2, MODEL_SAHANA_3
from api_keys import fetch_api_keys, get_next_key_index, KeyRotator
from database import get_recent_history, save_message, get_user_temp, save_memory, get_user_model
from markdown_parse import markdown_to_html, escape_html
from message import send_message, send_chat_action

logger = logging.getLogger("mero.api")
MAX_OUTPUT_TOKENS = 64000

MODEL_MAP = {
    "sahana-1": MODEL_SAHANA_1,
    "sahana-2": MODEL_SAHANA_2,
    "sahana-3": MODEL_SAHANA_3,
}

async def get_gemini_model(chat_id: int) -> str:
    m = await get_user_model(chat_id)
    return MODEL_MAP.get(m, MODEL_SAHANA_1)

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
    },
    {
        "name": "load_memory",
        "description": "Load and retrieve saved memories from long-term storage when context is needed.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
]

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

async def try_api_call(body_json: str, model: str) -> tuple[Optional[str], Optional[str]]:
    if not await fetch_api_keys(): return None, "No API keys available"
    start_idx = await get_next_key_index()
    rotator = KeyRotator(start_idx)
    last_error = None
    async with httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_connections=500, max_keepalive_connections=100)) as client:
        while True:
            key = rotator.get_next_key()
            if key is None: break
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            try:
                resp = await client.post(url, content=body_json, headers={"Content-Type": "application/json"})
                if resp.status_code == 200: return resp.text, None
                error_text = resp.text
                logger.warning(f"API call failed with status {resp.status_code}: {error_text}")
                if resp.status_code == 429:
                    await asyncio.sleep(0.5)
                    last_error = f"Status 429: Rate limit"
                    continue
                if resp.status_code >= 500:
                    await asyncio.sleep(1.0)
                    last_error = f"Status {resp.status_code}: Server error"
                    continue
                return None, f"Status {resp.status_code}: {error_text}"
            except Exception as exc:
                logger.warning(f"API call exception: {exc}")
                last_error = str(exc)
                await asyncio.sleep(1.0)
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
        if use_tools: tools.append({"google_search": {}})
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
    elif func_name == "load_memory":
        from database import get_memories
        memories = await get_memories(cid)
        if memories:
            return {"status": "success", "memories": memories, "message": f"Loaded {len(memories)} memories"}
        return {"status": "success", "memories": [], "message": "No memories found"}
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
    if not await fetch_api_keys(): return None
    model = await get_gemini_model(cid)
    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": _normalize_parts(parts)}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.4},
    }
    content, err = await try_api_call(json.dumps(body), model)
    if not content: return None
    text, _ = extract_ai_text(content)
    return text

async def handle_gemini(cid: int, current_parts: list, system_text: str, use_tools: bool = True, use_functions: bool = True, user_name: str = "User") -> Optional[str]:
    model = await get_gemini_model(cid)
    history = await get_recent_history(cid, CONTEXT_SIZE)
    body = build_body(history, current_parts, system_text, use_tools, use_functions)
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