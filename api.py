import json
import logging
import httpx
import asyncio
from typing import Optional
from config import CONTEXT_SIZE, MODEL_SAHANA_1, MODEL_SAHANA_2, MODEL_SAHANA_3
from api_keys import fetch_api_keys, get_next_key_index, KeyRotator
from database import get_recent_history, save_message, get_user_temp, save_memory, get_user_model, get_user_tools, get_file_data
from markdown_parse import markdown_to_html, escape_html
from message import send_message, send_chat_action

logger = logging.getLogger("mero.api")
MAX_OUTPUT_TOKENS = 64000
MAX_FUNCTION_CALL_TURNS = 8
WEB_SEARCH_MODEL = "gemini-2.5-flash"

MODEL_MAP = {"sahana-1": MODEL_SAHANA_1, "sahana-2": MODEL_SAHANA_2, "sahana-3": MODEL_SAHANA_3}

async def get_gemini_model(chat_id: int) -> str:
    m = await get_user_model(chat_id)
    return MODEL_MAP.get(m, MODEL_SAHANA_1)

FUNCTION_DECLARATIONS = [
    {"name": "save_memory", "description": "Save an important piece of information about the user to long-term memory.", "parameters": {"type": "object", "properties": {"memory": {"type": "string", "description": "The information to save."}}, "required": ["memory"]}},
    {"name": "load_memory", "description": "Load and retrieve saved memories from long-term storage when context is needed.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "create_pdf", "description": "Create a PDF document from AI-generated content.", "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "The topic or instructions for the PDF."}, "file_name": {"type": "string", "description": "Optional PDF file name, without or with .pdf extension."}}, "required": ["topic"]}},
    {"name": "generate_image", "description": "Generate an AI image based on a text prompt.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed description of the image."}}, "required": ["prompt"]}},
    {"name": "web_search", "description": "Search the web in depth for current or factual information. Use this for latest/current topics, news, links, research, and fact checks.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query or topic."}}, "required": ["query"]}},
    {"name": "text_to_speech", "description": "Convert text to speech audio and send it to the user as a voice message.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to speak."}}, "required": ["text"]}},
    {"name": "translate_text", "description": "Translate text into a target language.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}},
    {"name": "summarize_text", "description": "Summarize long text into concise key points.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "analyze_text", "description": "Analyze text statistics such as words, characters, and paragraphs.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
]

GEMINI_SUPPORTED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/gif", "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/opus", "audio/flac", "audio/aac", "audio/webm", "audio/mp4", "audio/m4a", "video/mp4", "video/webm", "video/quicktime", "video/x-matroska", "video/x-msvideo", "video/3gpp", "application/pdf", "text/plain", "text/html", "text/css", "text/javascript", "text/csv", "text/xml", "application/json", "application/xml", "text/markdown"}

def normalize_mime_type(mime: str) -> str:
    mime = (mime or "").strip().lower()
    if mime in GEMINI_SUPPORTED_MIMES: return mime
    if mime.startswith("text/") or "javascript" in mime or "json" in mime or "xml" in mime: return "text/plain"
    return "application/octet-stream"

def _normalize_part_keys(part: dict) -> dict:
    def _compact(data: dict) -> dict: return {k: v for k, v in data.items() if v not in ("", None)}
    if "fileData" in part and isinstance(part["fileData"], dict):
        fd = part["fileData"]
        return {"file_data": _compact({"mime_type": normalize_mime_type(fd.get("mimeType") or fd.get("mime_type")), "file_uri": fd.get("fileUri") or fd.get("uri") or fd.get("file_uri")})}
    if "file_data" in part and isinstance(part["file_data"], dict):
        fd = part["file_data"]
        return {"file_data": _compact({"mime_type": normalize_mime_type(fd.get("mime_type") or fd.get("mimeType")), "file_uri": fd.get("file_uri") or fd.get("uri") or fd.get("fileUri")})}
    if "inlineData" in part and isinstance(part["inlineData"], dict):
        ind = part["inlineData"]
        return {"inline_data": _compact({"mime_type": normalize_mime_type(ind.get("mimeType") or ind.get("mime_type")), "data": ind.get("data")})}
    if "inline_data" in part and isinstance(part["inline_data"], dict):
        ind = part["inline_data"]
        return {"inline_data": _compact({"mime_type": normalize_mime_type(ind.get("mime_type") or ind.get("mimeType")), "data": ind.get("data")})}
    if "text" in part:
        text_val = (part.get("text") or "").strip()
        return {"text": text_val} if text_val else {}
    return part

def _normalize_parts(parts: list) -> list:
    return [p for p in (_normalize_part_keys(part) for part in parts if isinstance(part, dict)) if p]

async def try_api_call(body_json: str, model: str) -> tuple[Optional[str], Optional[str]]:
    if not await fetch_api_keys(): return None, "No API keys available"
    rotator = KeyRotator(await get_next_key_index()); tried_keys=[]; last_error=None
    async with httpx.AsyncClient(timeout=180.0, limits=httpx.Limits(max_connections=500, max_keepalive_connections=100)) as client:
        while True:
            key_idx, key = rotator.get_next_key(tried_keys)
            if key is None: break
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            try:
                resp = await client.post(url, content=body_json, headers={"Content-Type":"application/json"})
                if resp.status_code == 200: return resp.text, None
                last_error = f"Status {resp.status_code}: {resp.text}"; logger.warning("Gemini key index %s failed: %s", key_idx, last_error)
                if resp.status_code == 429 or resp.status_code >= 500: await asyncio.sleep(0.5)
            except Exception as exc:
                last_error = str(exc); logger.warning("Gemini key index %s exception: %s", key_idx, exc); await asyncio.sleep(0.5)
    return None, f"All API keys exhausted. Tried key indexes: {tried_keys}. Last error: {last_error or 'unknown'}"

def build_body(history_messages: list[dict], current_parts: list, system_text: str, tool_prefs: dict | None = None, use_functions: bool = True) -> dict:
    raw_msgs=[]
    for msg in history_messages:
        text=(msg.get("text") or "").strip()
        if text: raw_msgs.append({"role":"user" if msg.get("role")=="user" else "model", "parts":[{"text":text}]})
    norm=_normalize_parts(current_parts)
    if norm: raw_msgs.append({"role":"user","parts":norm})
    contents=[]
    for msg in raw_msgs:
        if contents and contents[-1]["role"] == msg["role"]: contents[-1]["parts"].extend(msg["parts"])
        else: contents.append(msg)
    while contents and contents[0]["role"] != "user": contents.pop(0)
    if not contents: contents.append({"role":"user","parts":norm or [{"text":"Hello"}]})
    body={"systemInstruction":{"parts":[{"text":system_text}]},"contents":contents,"generationConfig":{"maxOutputTokens":MAX_OUTPUT_TOKENS,"temperature":1.0}}
    if use_functions and (tool_prefs or {}).get("web_search", True): body["tools"]=[{"functionDeclarations":FUNCTION_DECLARATIONS}]
    elif use_functions: body["tools"]=[{"functionDeclarations":[f for f in FUNCTION_DECLARATIONS if f["name"] != "web_search"]}]
    return body

def extract_sources(data: dict) -> list[dict]:
    sources=[]; seen=set()
    try:
        for chunk in data.get("candidates", [{}])[0].get("groundingMetadata", {}).get("groundingChunks", []):
            web=chunk.get("web", {}); uri=web.get("uri", ""); title=web.get("title", "Source")
            if uri and uri not in seen: seen.add(uri); sources.append({"title":title.strip(),"url":uri.strip()})
    except Exception: pass
    return sources

def extract_ai_text(content: str) -> tuple[str, list[dict]]:
    try: data=json.loads(content)
    except json.JSONDecodeError: return "Failed to parse AI response.", []
    candidates=data.get("candidates", [])
    if not candidates: return "No response received from AI.", []
    parts=candidates[0].get("content", {}).get("parts", [])
    ai_text="\n".join(p.get("text", "") for p in parts if p.get("text"))
    return (ai_text.strip() or "No response received from AI."), extract_sources(data)

def extract_function_calls(content: str) -> list[dict]:
    try: data=json.loads(content)
    except json.JSONDecodeError: return []
    calls=[]
    for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        fc=part.get("functionCall")
        if fc: calls.append({"name":fc.get("name", ""), "args":fc.get("args", {}) or {}})
    return calls

def format_response_with_sources(ai_text: str, sources: list[dict]) -> str:
    html=markdown_to_html(ai_text)
    if sources:
        html += "\n\n📌 <b>Sources:</b>\n" + "".join(f'• <a href="{escape_html(s["url"])}">{escape_html(s["title"])}</a>\n' for s in sources)
    return html

async def attachment_reply_markup(cid: int) -> dict | None:
    if await get_file_data(cid):
        from settings import ikb, btn
        return ikb([[btn("🧹 Remove attachment", "cancel_attachment")]])
    return None

async def web_search(query: str, cid: int = 0) -> dict:
    system = "You are a web searcher AI. Your task is to search the web on given topic in depth step by step. You've to cover all aspects of the topic to make results better. Return the full body of the search results. Format in a better way. Search at least 20 pages for each topic."
    body={"systemInstruction":{"parts":[{"text":system}]},"contents":[{"role":"user","parts":[{"text":query}]}],"tools":[{"google_search":{}}],"generationConfig":{"maxOutputTokens":MAX_OUTPUT_TOKENS,"temperature":0.2}}
    content, err = await try_api_call(json.dumps(body), WEB_SEARCH_MODEL)
    if not content: return {"status":"failed","message":err or "Web search failed"}
    text, sources = extract_ai_text(content)
    return {"status":"success","query":query,"results":text,"sources":sources,"message":"Web search completed successfully."}

async def _execute_function(cid: int, func_name: str, args: dict, user_name: str = "User") -> dict:
    try:
        if func_name == "save_memory":
            memory_text = (args.get("memory") or "").strip()
            if memory_text:
                await save_memory(cid, memory_text)
                return {"status":"success","message":"Memory updated successfully.","memory":memory_text}
        if func_name == "load_memory":
            from database import get_memories
            memories=await get_memories(cid)
            return {"status":"success","memories":memories,"message":f"Loaded {len(memories)} memories."}
        if func_name == "create_pdf":
            from texttopdf import execute_text_to_pdf
            topic=(args.get("topic") or "").strip(); file_name=(args.get("file_name") or "").strip() or None
            if topic:
                ok = await execute_text_to_pdf(cid, topic, file_name=file_name, announce=False)
                return {"status":"success" if ok else "failed","message":"PDF created successfully." if ok else "PDF creation failed."}
        if func_name == "generate_image":
            from image_generation import execute_image
            prompt=(args.get("prompt") or "").strip()
            if prompt:
                ok = await execute_image(cid, prompt, user_name, announce=False)
                return {"status":"success" if ok else "failed","message":"Image generated successfully." if ok else "Image generation failed."}
        if func_name == "web_search": return await web_search((args.get("query") or "").strip(), cid)
        if func_name == "text_to_speech":
            from tts import generate_tts
            from message import send_voice_bytes
            audio = await generate_tts((args.get("text") or "").strip())
            if audio:
                await send_voice_bytes(cid, audio, None, "speech.mp3", "audio/mpeg")
                return {"status":"success","message":"Text converted to speech successfully."}
        if func_name == "translate_text":
            from tools import run_text_translator, resolve_language
            lang_code, lang_name = resolve_language(args.get("target_language", ""))
            if lang_code and lang_name:
                await run_text_translator(cid, args.get("text", ""), lang_code, lang_name)
                return {"status":"success","message":"Text translated successfully."}
        if func_name == "summarize_text":
            from tools import run_content_summarizer
            await run_content_summarizer(cid, args.get("text", "")); return {"status":"success","message":"Text summarized successfully."}
        if func_name == "analyze_text":
            from tools import run_text_analyzer
            await run_text_analyzer(cid, args.get("text", "")); return {"status":"success","message":"Text analyzed successfully."}
    except Exception as exc:
        logger.exception("function_execution_failed name=%s", func_name)
        return {"status":"failed","message":f"{func_name} failed: {exc}"}
    return {"status":"failed","message":f"Unknown function or missing required arguments: {func_name}"}

async def _send_function_response(cid: int, model: str, body: dict, function_calls: list[dict], user_name: str, depth: int = 0) -> tuple[Optional[str], list[dict]]:
    if depth >= MAX_FUNCTION_CALL_TURNS: return "I completed the tool steps, but hit the maximum tool-call limit before I could finish the final reply.", []
    contents=list(body.get("contents", [])); contents.append({"role":"model","parts":[{"functionCall":{"name":fc["name"],"args":fc["args"]}} for fc in function_calls]})
    fr_parts=[]
    for fc in function_calls:
        result=await _execute_function(cid, fc["name"], fc["args"], user_name=user_name)
        fr_parts.append({"functionResponse":{"name":fc["name"],"response":result}})
    contents.append({"role":"user","parts":fr_parts})
    follow_up=dict(body); follow_up["contents"]=contents
    content, _ = await try_api_call(json.dumps(follow_up), model)
    if not content:
        messages=[r["functionResponse"]["response"].get("message", "Tool finished.") for r in fr_parts]
        return " ".join(messages), []
    more=extract_function_calls(content)
    if more: return await _send_function_response(cid, model, follow_up, more, user_name, depth+1)
    return extract_ai_text(content)

async def call_gemini_raw(cid: int, parts: list, system_text: str, model_override: str | None = None) -> Optional[str]:
    model = model_override or await get_gemini_model(cid)
    body={"systemInstruction":{"parts":[{"text":system_text}]},"contents":[{"role":"user","parts":_normalize_parts(parts)}],"generationConfig":{"maxOutputTokens":MAX_OUTPUT_TOKENS,"temperature":0.4}}
    content, _ = await try_api_call(json.dumps(body), model)
    if not content: return None
    text, _ = extract_ai_text(content)
    return None if text in ("No response received from AI.", "Failed to parse AI response.") else text

async def handle_gemini(cid: int, current_parts: list, system_text: str, use_tools: bool = True, use_functions: bool = True, user_name: str = "User") -> Optional[str]:
    model=await get_gemini_model(cid); history=await get_recent_history(cid, CONTEXT_SIZE); user_tools=await get_user_tools(cid)
    body=build_body(history, current_parts, system_text, user_tools if use_tools else {}, use_functions=use_functions)
    body["generationConfig"]["temperature"] = await get_user_temp(cid)
    content, err = await try_api_call(json.dumps(body), model)
    if content:
        calls=extract_function_calls(content)
        if calls:
            for fc in calls:
                await send_chat_action(cid, {"create_pdf":"upload_document","generate_image":"upload_photo"}.get(fc["name"], "typing"))
            final_text, sources = await _send_function_response(cid, model, body, calls, user_name)
            if not final_text or final_text in ("No response received from AI.", "Failed to parse AI response."):
                final_text = "Done — I completed that for you."
            await save_message(cid, "model", final_text)
            await send_message(cid, format_response_with_sources(final_text, sources), parse_mode="HTML", reply_markup=await attachment_reply_markup(cid))
            return final_text
        ai_text, sources=extract_ai_text(content)
        if ai_text in ("No response received from AI.", "Failed to parse AI response."):
            ai_text="I couldn't generate a text response. Please try again."
        await save_message(cid,"model",ai_text); await send_message(cid, format_response_with_sources(ai_text, sources), parse_mode="HTML", reply_markup=await attachment_reply_markup(cid)); return ai_text
    error=f"Error: {err or 'Unknown error occurred'}"; await save_message(cid,"model",error); await send_message(cid,error); return None
