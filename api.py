import json
import base64
import logging
import asyncio
import time
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONTEXT_SIZE, MODEL_SAHANA_1, MODEL_SAHANA_2, MODEL_SAHANA_3, DEFAULT_MODEL, WEB_SEARCH_MODEL
from api_keys import (
    fetch_api_keys, get_next_key_index, _gemini_request, HTTPException,
    normalize_mime_type, upload_file_to_gemini, upload_file_with_retry, delete_gemini_file,
    is_key_on_cooldown, mark_key_error, clear_key_error,
    compute_backoff_delay, get_rate_limiter,
)
from database import get_recent_history, save_message, get_user_temp, save_memory, save_memories_batch, get_memories, get_user_model, format_memories_block, get_formatted_memories
from markdown_parse import markdown_to_html, escape_html
from message import send_message, send_chat_action

logger = logging.getLogger("mero.api")
MAX_OUTPUT_TOKENS = 64000
MAX_INLINE_FILE_BYTES = 2 * 1024 * 1024  # 2MB threshold for Files API
MAX_FUNCTION_CALL_TURNS = 6
MAX_CONCURRENT_WORKERS = 50

# Strict, vast system instruction for the dedicated web_search helper model (gemini-2.5-flash + google_search)
WEB_SEARCH_SYSTEM_INSTRUCTION = (
    "You are a STRICT, VAST, and HIGHLY ACCURATE Web Search Specialist powered by Google Search grounding.\n"
    "Your SOLE mission is to search the web for the USER'S QUERY using the `google_search` tool and produce EXHAUSTIVE, STRUCTURED, DETAILED results.\n\n"
    "STRICT RULES:\n"
    "- You MUST use the google_search tool for every query — never answer from memory alone. If you fail to trigger google_search, your response is invalid.\n"
    "- Generate 1-3 optimized, diverse search queries covering all aspects, synonyms, and recent angles of the topic before synthesizing.\n"
    "- Search DEEPLY — consider equivalent to scanning at least 20 pages per topic, covering news, official sites, docs, and diverse sources.\n"
    "- Coverage must be COMPREHENSIVE: include definitions, recent developments, key facts, numbers, dates, prices, comparisons, pros/cons, opinions, and context.\n"
    "- NEVER hallucinate URLs, titles, or facts — only use what grounding returns. If data is insufficient, explicitly state the gap and suggest a refined query.\n"
    "- Be VAST and DETAILED: for EACH distinct search result, provide a dedicated section with granular information.\n"
    "- Prioritize recency, authority, and relevance. When conflicting data appears, note the discrepancy and cite both sources.\n"
    "- Output MUST be well-structured, clean markdown, easy to parse, with no excessive fluff, apologies, or meta commentary.\n\n"
    "REQUIRED OUTPUT FORMAT (strictly follow):\n"
    "For EACH result, output exactly:\n"
    "### [Result N: Title]\n"
    "- **Source:** Title — URL\n"
    "- **Summary:** 2-3 sentence concise snippet of what the page says about the query.\n"
    "- **Key Details:** Bullet list of 3-6 most important facts, numbers, quotes, or insights from that page.\n"
    "- **Relevance:** 1 sentence explaining why this result matters for the query.\n"
    "After all results (aim for 5-10 results minimum), provide:\n"
    "## Synthesis\n"
    "A 3-5 sentence overall answer synthesizing across sources, highlighting consensus, key numbers, and final takeaway.\n"
    "Use markdown only. Keep each result detailed but scannable."
)

FUNCTION_DECLARATIONS = [
    {
        "name": "save_memory",
        "description": "Save important personal facts, preferences, or details about the user to long-term memory for future recall. Use when user shares ANY durable personal info — name, age, location, birthday, profession, hobbies, goals, likes/dislikes, projects, or says 'remember this'. SUPPORTS BULK: you can save multiple distinct facts in a single call via `memories` array (preferred when user shares 2+ facts). Each array element must be one concise, self-contained fact (e.g., 'User name is Sujan Rai', 'User lives in Dhankuta district, Nepal', 'User works at an engineering company'). If only one fact, you may use `memory` string. Use EITHER `memory` (single) OR `memories` (array) — `memories` is preferred for multiple facts to prevent multiple tool calls. After saving, you MUST still directly answer the user's original request (e.g., provide requested Python code) in your final response — do NOT stop after saving.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The user ID of the current user. Must be the system-provided user_id."
                },
                "memory": {
                    "type": "string",
                    "description": "Single memory to save (concise fact, e.g., 'User is a Python developer from Nepal'). Use when only ONE fact to save. Ignored if `memories` is provided."
                },
                "memories": {
                    "type": "array",
                    "description": "Array of memory strings to save in bulk — PREFERRED when user shares 2+ distinct facts. Each element is one concise, self-contained fact. Example: ['User name is Sujan Rai', 'User lives in Dhankuta, Nepal', 'User works at an engineering company']. Max 10 per call, each <=1000 chars.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "load_memory",
        "description": "Load and retrieve saved long-term memories for the user when you need context, facts, or recall about the user. Use before answering personal questions or when prior context would improve accuracy.",
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
        "description": "Create a downloadable PDF document on a given topic. Use when user explicitly requests a PDF, document, or export.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or subject content for the PDF document."
                },
                "file_name": {
                    "type": "string",
                    "description": "Optional filename for the PDF (e.g., 'report.pdf')."
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "generate_image",
        "description": "Generate an AI image based on a text prompt. Use when user asks to create, draw, generate, or imagine an image, logo, or illustration.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A detailed description of the image to generate, including style, colors, composition."
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the live web for real-time, verified information using Google Search grounding. Use for current events, news, prices, recent facts, definitions requiring freshness, or any query beyond knowledge cutoff. The query should be a clear, specific natural-language question or keywords (e.g., 'latest iPhone 16 price in Nepal 2026', 'who won Champions League 2025').",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web. Clear, specific, and concise (max 500 chars)."
                }
            },
            "required": ["query"]
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
    if "fileData" in part and isinstance(part["fileData"], dict):
        fd = part["fileData"]
        normalized = _compact({"mimeType": normalize_mime_type(fd.get("mimeType") or fd.get("mime_type") or "application/octet-stream"), "fileUri": fd.get("fileUri") or fd.get("file_uri") or fd.get("uri")})
        return {"fileData": normalized} if normalized else {}
    if "file_data" in part and isinstance(part["file_data"], dict):
        fd = part["file_data"]
        normalized = _compact({"mimeType": normalize_mime_type(fd.get("mime_type") or fd.get("mimeType")), "fileUri": fd.get("fileUri") or fd.get("file_uri") or fd.get("uri")})
        return {"fileData": normalized} if normalized else {}
    if "fileUri" in part:
        mime = part.get("mimeType") or part.get("mime_type") or "application/octet-stream"
        return {"fileData": _compact({"mimeType": normalize_mime_type(mime), "fileUri": part.get("fileUri")})}
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
        m = re.search(r'"retryDelay"\s*:\s*"([^"]+)"', raw_msg)
        if m:
            delay = m.group(1)
            return f"⏳ All API keys hit quota limits. Please retry in {delay}. (Automatic rotation exhausted all keys)"
        if "retry" in low:
            return "⏳ All API keys are rate-limited right now. Please wait ~30-60 seconds and try again. (Automatic key rotation tried all available keys)"
        return "⏳ Service is busy (quota exceeded). Please try again in ~30 seconds. All keys were rotated automatically."
    if "503" in raw_msg or "unavailable" in low or "overloaded" in low:
        return "⚠️ Gemini service is temporarily overloaded. Please try again in a few seconds."
    if "500" in raw_msg:
        return "⚠️ Gemini service error. Please try again shortly."
    if "401" in raw_msg or "403" in raw_msg or "permission" in low:
        return "⚠️ Some API keys are invalid or have permission issues. The system rotated to next keys but none succeeded."
    short = raw_msg[:500]
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
    low = raw_msg.lower() if isinstance(raw_msg, str) else ""
    if "429" in raw_msg or "resource_exhausted" in low or "quota" in low:
        try:
            from api_keys import time_until_next_key_available, get_available_keys
            wait = time_until_next_key_available()
            if 0 < wait <= 12.0:
                logger.info(f"All keys 429, waiting {wait:.1f}s for next key then retrying once")
                await asyncio.sleep(wait + 0.4)
                start_idx2 = await get_next_key_index()
                data2, err2 = await _gemini_request(url, body, start_idx2)
                if data2:
                    return data2, None
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
    if use_functions:
        body["tools"] = [{"functionDeclarations": FUNCTION_DECLARATIONS}]
        body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return body

def extract_sources(data: dict) -> list[dict]:
    """Extract grounding sources VERY correctly from Gemini groundingMetadata.

    Handles both generateContent and Files API shapes.
    Dedupes by URL, preserves order, handles edge cases.
    """
    sources: list[dict] = []
    seen: set[str] = set()
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            return []
        cand = candidates[0]
        gm = cand.get("groundingMetadata") or cand.get("grounding_metadata") or {}
        # groundingChunks
        chunks = gm.get("groundingChunks") or gm.get("grounding_chunks") or []
        for chunk in chunks:
            web = chunk.get("web") or chunk.get("Web") or {}
            if not isinstance(web, dict):
                continue
            uri = (web.get("uri") or web.get("url") or "").strip()
            title = (web.get("title") or web.get("name") or "Source").strip()
            if not uri:
                continue
            # Normalize uri
            if uri not in seen:
                seen.add(uri)
                # Clean title fallback to domain if empty
                if not title or title.lower() == "source":
                    try:
                        from urllib.parse import urlparse
                        title = urlparse(uri).netloc or title
                    except Exception:
                        pass
                sources.append({"title": title[:200], "url": uri})
        # Fallback: try retrievalMetadata / searchEntryPoint is not a source, ignore
        # Also handle alternative location: candidate may have groundingMetadata inside content?
        if not sources:
            # Try alternative nested path (some versions)
            alt = data.get("groundingMetadata") or {}
            chunks2 = alt.get("groundingChunks") or []
            for chunk in chunks2:
                web = chunk.get("web", {})
                uri = (web.get("uri") or "").strip()
                title = (web.get("title") or "Source").strip()
                if uri and uri not in seen:
                    seen.add(uri)
                    sources.append({"title": title, "url": uri})
    except Exception as exc:
        logger.debug(f"extract_sources failed: {exc}")
    return sources

def extract_ai_text(data: dict) -> tuple[str, list[dict]]:
    candidates = data.get("candidates", [])
    if not candidates: return "No response received from AI.", []
    parts = candidates[0].get("content", {}).get("parts", [])
    ai_text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    ai_text = ai_text.strip() if ai_text else "No response received from AI."
    sources = extract_sources(data)
    return ai_text, sources

def extract_function_calls(data: dict) -> list[dict]:
    candidates = data.get("candidates", [])
    if not candidates: return []
    parts = candidates[0].get("content", {}).get("parts", [])
    calls = []
    for part in parts:
        fc = part.get("functionCall")
        if fc:
            # Preserve id if present (Gemini 3 returns id)
            calls.append({"name": fc.get("name", ""), "args": fc.get("args", {}) or {}, "id": fc.get("id")})
    return calls

def format_sources_block(sources: list[dict]) -> str:
    """Format sources into a markdown/HTML block very correctly."""
    if not sources:
        return ""
    lines = ["\n📌 **Sources:**"]
    for idx, s in enumerate(sources, 1):
        title = escape_html(s.get("title", "Source"))
        url = s.get("url", "").strip()
        if url:
            # Escape url for HTML attribute
            safe_url = escape_html(url)
            lines.append(f"{idx}. <a href=\"{safe_url}\">{title}</a>")
        else:
            lines.append(f"{idx}. {title}")
    return "\n".join(lines)

def format_sources_markdown(sources: list[dict]) -> str:
    """Plain markdown version for model-internal formatted_sources."""
    if not sources:
        return ""
    lines = ["\n📌 Sources:"]
    for idx, s in enumerate(sources, 1):
        title = (s.get("title") or "Source").strip()
        url = s.get("url", "").strip()
        if url:
            lines.append(f"{idx}. {title} - {url}")
        else:
            lines.append(f"{idx}. {title}")
    return "\n".join(lines)

def format_response_with_sources(ai_text: str, sources: list[dict] = None) -> str:
    """Convert markdown ai_text to HTML and append formatted sources if present."""
    html = markdown_to_html(ai_text or "")
    if sources:
        html += "\n" + markdown_to_html(format_sources_markdown(sources)) if False else ""  # placeholder
        # Actually append HTML sources block
        html += format_sources_block(sources)
    return html

# =============================================================================
# WEB_SEARCH FUNCTION — Dedicated helper using gemini-2.5-flash + google_search
# Architecture: main model -> web_search(query) -> gemini-2.5-flash(google_search) -> formatted_ai_response + formatted_sources -> main model -> final answer
# All API calls use concurrent pool with max_workers=50 via ThreadPoolExecutor / asyncio semaphore
# =============================================================================

async def web_search(query: str, cid: int = 0) -> dict:
    """Search the web using grounded Gemini model.

    Uses model `gemini-2.5-flash` with system instruction that demands
    structured, detailed per-result output and the `google_search` tool.
    Returns dict with `formatted_output = f\"{formatted_ai_response}\\n{formatted_sources}\\n\"`
    pronounced exactly as required, ready to be returned to the main model.

    Concurrency: executed via the shared API pool (max_workers=50 internally via key rotation + rate limiter).
    """
    query_clean = (query or "").strip()
    if not query_clean:
        return {"status": "error", "message": "Query is required and cannot be empty.", "query": query}
    if len(query_clean) > 500:
        query_clean = query_clean[:500].strip()

    # Build grounded search body
    body: dict = {
        "systemInstruction": {"parts": [{"text": WEB_SEARCH_SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": query_clean}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.2},
    }

    data, err = await try_api_call(WEB_SEARCH_MODEL, body)
    if not data:
        logger.warning(f"web_search failed query='{query_clean[:60]}' err={err}")
        return {"status": "failed", "message": err or "Web search failed", "query": query_clean, "results": "", "sources": [], "formatted_output": ""}

    # Extract very correctly
    formatted_ai_response, sources = extract_ai_text(data)

    # Handle empty edge case
    if not formatted_ai_response or formatted_ai_response in ("No response received from AI.", "Failed to parse AI response."):
        formatted_ai_response = f"No detailed search results found for query: {query_clean}. Please try a more specific query."

    # Format sources very correctly (deduplicated, verified)
    formatted_sources = format_sources_markdown(sources)
    if not formatted_sources and sources:
        # Fallback formatting
        formatted_sources = "\n".join(f"{s['title']} - {s['url']}" for s in sources)
    elif not formatted_sources:
        formatted_sources = "\n📌 Sources: (grounding provided but no explicit chunks — see synthesis above)"

    # Required output to return to the model: f"{formatted_ai_response}\n{formatted_sources}\n"
    formatted_output = f"{formatted_ai_response}\n{formatted_sources}\n"

    # Also prepare HTML block for potential direct display (not used for functionResponse but useful)
    html_sources = format_sources_block(sources)

    logger.info(f"web_search success query='{query_clean[:60]}' sources={len(sources)}")

    return {
        "status": "success",
        "message": "Web search completed successfully.",
        "query": query_clean,
        "results": formatted_ai_response,
        "sources": sources,
        "formatted_sources": formatted_sources,
        "html_sources": html_sources,
        "formatted_output": formatted_output,
        # Also include combined for convenience per spec
        "combined": formatted_output,
    }


async def _execute_function(cid: int, func_name: str, args: dict, user_name: str = "User") -> dict:
    """Execute a single function call and return its response dict (for functionResponse).

    Handles save_memory, load_memory, create_pdf, generate_image, web_search.
    All functions are designed to work very correctly and be concurrency-safe.
    """
    try:
        # --- SAVE MEMORY — BULK CAPABLE ---
        if func_name == "save_memory":
            uid = args.get("user_id", cid)
            try:
                uid_int = int(uid)
            except Exception:
                uid_int = cid
            # Support both `memory` (single string) and `memories` (array) — bulk path preferred
            memories_arg = args.get("memories")
            memory_text = (args.get("memory") or "").strip() if isinstance(args.get("memory"), str) else ""
            batch: list[str] = []
            if isinstance(memories_arg, list) and memories_arg:
                for item in memories_arg:
                    if item is None:
                        continue
                    s = str(item).strip()
                    if s:
                        batch.append(s)
            if memory_text:
                batch.append(memory_text)
            # Also handle legacy comma/newline delimited single string that actually contains multiple facts?
            # We keep as single unless array given; model instructed to use array for multiples.
            if not batch:
                return {"status": "error", "message": "Memory text is required: provide `memory` (string) or `memories` (array of strings). Each fact should be concise."}
            # Deduplicate within batch case-insensitively before saving (preserve order)
            seen = set()
            deduped_batch: list[str] = []
            for m in batch:
                low = m.lower()
                if low not in seen:
                    seen.add(low)
                    deduped_batch.append(m)
            if len(deduped_batch) == 1:
                # Single-memory path — keep simple response for backward compat
                single = deduped_batch[0]
                saved = await save_memory(uid_int, single)
                if saved:
                    return {"status": "success", "message": f"Memory saved successfully: {single}", "memory": single, "memories": [single], "user_id": uid_int, "saved_count": 1}
                else:
                    memories = await get_memories(uid_int)
                    if any(single.lower() == mm.lower() for mm in memories):
                        return {"status": "success", "message": f"Memory already exists (duplicate not saved): {single}", "memory": single, "memories": [single], "user_id": uid_int, "saved_count": 0, "duplicate": True}
                    return {"status": "error", "message": "Failed to save memory (storage error).", "memory": single}
            # Bulk path: save N memories concurrently via batch helper
            result = await save_memories_batch(uid_int, deduped_batch)
            saved_items = result.get("saved_items", [])
            dup_items = result.get("duplicate_items", [])
            total = result.get("total", 0)
            saved_cnt = result.get("saved", 0)
            dup_cnt = result.get("duplicates", 0)
            if saved_cnt > 0:
                # Build concise but informative message that does NOT mislead main model to stop
                msg = f"Memories updated: {saved_cnt}/{total} new saved"
                if dup_cnt:
                    msg += f", {dup_cnt} duplicates skipped"
                msg += f". Saved: {', '.join(saved_items[:5])}{'...' if len(saved_items) > 5 else ''}"
                # CRITICAL: instruct main model to continue original task
                msg += " — Now continue and fully answer the user's original request using these saved facts where relevant."
                return {
                    "status": "success",
                    "message": msg,
                    "memories": saved_items,
                    "saved_items": saved_items,
                    "duplicate_items": dup_items,
                    "user_id": uid_int,
                    "saved_count": saved_cnt,
                    "duplicate_count": dup_cnt,
                    "total": total,
                }
            else:
                # All duplicates
                if dup_cnt > 0 and result.get("failed", 0) == 0:
                    return {
                        "status": "success",
                        "message": f"All {dup_cnt} memories already existed (duplicates not saved). Continue answering the user's original request.",
                        "memories": deduped_batch,
                        "duplicate_items": dup_items,
                        "user_id": uid_int,
                        "saved_count": 0,
                        "duplicate_count": dup_cnt,
                    }
                return {"status": "error", "message": "Failed to save memories (storage error).", "memories": deduped_batch}
        
        # --- LOAD MEMORY ---
        elif func_name == "load_memory":
            uid = args.get("user_id", cid)
            try:
                uid_int = int(uid)
            except Exception:
                uid_int = cid
            memories = await get_memories(uid_int)
            formatted = format_memories_block(memories) if memories else "No saved memories found for this user."
            # Return formatted manner as requested
            if memories:
                return {
                    "status": "success",
                    "message": f"Loaded {len(memories)} saved memories.",
                    "user_id": uid_int,
                    "memories": memories,
                    "formatted": formatted,
                    "count": len(memories)
                }
            return {
                "status": "success",
                "message": "No saved memories found for this user.",
                "user_id": uid_int,
                "memories": [],
                "formatted": formatted,
                "count": 0
            }
        
        # --- CREATE PDF ---
        elif func_name == "create_pdf":
            topic = (args.get("topic") or "").strip()
            file_name = (args.get("file_name") or "").strip() or None
            if not topic:
                return {"status": "error", "message": "Topic is required for PDF creation."}
            from texttopdf import execute_text_to_pdf
            # execute_text_to_pdf handles sending to Telegram and returns bool
            ok = await execute_text_to_pdf(cid, topic, file_name=file_name, announce=False)
            if ok:
                return {"status": "success", "message": f"PDF created successfully for topic: {topic}", "topic": topic}
            return {"status": "failed", "message": f"PDF creation failed for topic: {topic}. Please refine the topic and try again.", "topic": topic}
        
        # --- GENERATE IMAGE ---
        elif func_name == "generate_image":
            prompt = (args.get("prompt") or "").strip()
            if not prompt:
                return {"status": "error", "message": "Prompt is required for image generation."}
            from image_generation import execute_image
            ok = await execute_image(cid, prompt, user_name, announce=False)
            if ok:
                return {"status": "success", "message": f"Image generated successfully for: {prompt}", "prompt": prompt}
            return {"status": "failed", "message": "Image generation failed. Please try again with a different prompt.", "prompt": prompt}
        
        # --- WEB SEARCH ---
        elif func_name == "web_search":
            query = (args.get("query") or "").strip()
            if not query:
                return {"status": "error", "message": "Query is required for web search and cannot be empty."}
            result = await web_search(query, cid)
            # web_search already returns status/message etc; pass through as functionResponse
            # Ensure response is JSON-serializable and contains formatted_output
            return result

    except Exception as exc:
        logger.exception(f"_execute_function failed name={func_name} cid={cid}")
        return {"status": "failed", "message": f"Function {func_name} encountered an error: {exc}"}
    
    return {"status": "error", "message": f"Unknown function or missing required arguments: {func_name}"}


async def _execute_functions_concurrently(cid: int, function_calls: list[dict], user_name: str = "User") -> list[dict]:
    """Execute multiple function calls concurrently with max_workers=50.

    Uses asyncio.gather with semaphore to limit concurrency to MAX_CONCURRENT_WORKERS.
    Preserves order of results to match input order (required for functionResponse mapping).
    """
    if not function_calls:
        return []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

    async def _run_one(fc: dict) -> dict:
        async with semaphore:
            result = await _execute_function(cid, fc["name"], fc.get("args", {}) or {}, user_name=user_name)
            # Preserve id if provided (Gemini 3)
            resp = {"functionResponse": {"name": fc["name"], "response": result}}
            if fc.get("id"):
                resp["functionResponse"]["id"] = fc["id"]
            return resp

    # Gather concurrently, preserve order
    tasks = [_run_one(fc) for fc in function_calls]
    results = await asyncio.gather(*tasks)
    return results


async def _send_function_response(cid: int, model: str, body: dict, function_calls: list[dict], user_name: str = "User", depth: int = 0) -> Optional[str]:
    """Handle functionResponse round-trip, concurrently executing functions.

    Implements compositional calling: if the follow-up also contains functionCalls, recurse.
    Limits depth to MAX_FUNCTION_CALL_TURNS to prevent infinite loops.
    """
    if depth >= MAX_FUNCTION_CALL_TURNS:
        logger.warning(f"Max function call turns exceeded depth={depth} cid={cid}")
        return "I completed the tool steps but reached the maximum tool-call limit before finishing. Please try rephrasing or breaking the request into parts."

    contents = list(body.get("contents", []))
    # Append model turn with functionCalls
    fc_parts = []
    for fc in function_calls:
        part = {"functionCall": {"name": fc["name"], "args": fc.get("args", {}) or {}}}
        if fc.get("id"):
            part["functionCall"]["id"] = fc["id"]
        fc_parts.append(part)
    contents.append({"role": "model", "parts": fc_parts})
    
    # Execute all functions concurrently with max_workers=50
    fr_parts = await _execute_functions_concurrently(cid, function_calls, user_name=user_name)
    contents.append({"role": "user", "parts": fr_parts})
    
    follow_up = dict(body)
    follow_up["contents"] = contents
    # Preserve toolConfig and systemInstruction
    # follow_up already has them from body
    data, err = await try_api_call(model, follow_up)
    if not data:
        # If follow-up fails, return the function results as fallback so user still gets value
        # CRITICAL FIX: prioritize detailed outputs (formatted_output/results/formatted) over bare message
        # so user gets actual search results / loaded memories, not just "Web search completed successfully"
        logger.warning(f"_send_function_response follow-up failed cid={cid} err={err}")
        try:
            messages = []
            for part in fr_parts:
                resp = part.get("functionResponse", {}).get("response", {})
                fname = part.get("functionResponse", {}).get("name", "")
                # Priority order: detailed content first, then generic message
                # For web_search: formatted_output / combined / results
                # For load_memory: formatted
                # For save_memory: message (which already contains saved items summary)
                msg = (
                    resp.get("formatted_output")
                    or resp.get("combined")
                    or resp.get("results")
                    or resp.get("formatted")
                    or resp.get("message")
                    or ""
                )
                if msg:
                    # For web_search, msg already contains full formatted_ai_response + sources; wrap nicely
                    if fname == "web_search" and resp.get("status") == "success" and resp.get("formatted_output"):
                        messages.append(msg)
                    elif fname == "load_memory" and resp.get("formatted"):
                        messages.append(f"🧠 Loaded memories:\n{resp.get('formatted')}")
                    elif fname == "save_memory" and resp.get("status") == "success":
                        # Include save confirmation — but also note we couldn't generate final answer
                        messages.append(msg + "\n\n⚠️ Final synthesis failed due to API limit — please retry your original request and I'll include these memories in the answer.")
                    else:
                        messages.append(msg)
            fallback = "\n\n---\n\n".join(messages) if messages else "I finished the requested tool actions, but couldn't generate a final summary due to a temporary API issue. Please retry your request in a few seconds."
            return fallback
        except Exception:
            return None
    
    more_calls = extract_function_calls(data)
    if more_calls:
        return await _send_function_response(cid, model, follow_up, more_calls, user_name=user_name, depth=depth+1)
    
    ai_text, sources = extract_ai_text(data)
    # If no text but sources exist, still return with sources formatting
    if not ai_text or ai_text in ("No response received from AI.", "Failed to parse AI response."):
        ai_text = "Done — I completed that for you."
    # Append sources handling is done by caller; here return ai_text and let caller format
    # For backward compat, we return text; caller will handle sources if needed
    # But we should also store sources in a way that caller can use: we return tuple via side? For now handle in handle_gemini directly.
    # To preserve sources for caller, we could return both — however Python typing says Optional[str]. We'll just return text and handle sources via global? Instead we return text; caller will re-extract? Safer to return text and include sources in text? We'll handle differently.
    # We'll return text and caller will extract again? But we already have sources.
    # To pass sources, we could set an attribute on function? Simpler: just return text with sources formatted appended if present.
    if sources:
        ai_text = f"{ai_text}\n{format_sources_markdown(sources)}\n"
    return ai_text

async def call_gemini_raw(cid: int, parts: list, system_text: str, model_override: str | None = None) -> Optional[str]:
    model = model_override or await get_gemini_model(cid)
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
    return text if text and text not in ("No response received from AI.", "Failed to parse AI response.") else None


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

        # Already a fileUri/fileData -> normalize to fileData? Keep as is but normalize keys
        if part.get("fileData") or part.get("fileUri") or part.get("file_data"):
            # Normalize via _normalize_part_keys
            normalized = _normalize_part_keys(part)
            processed.append(normalized)
            continue

        processed.append(part)
    return processed



async def handle_gemini(cid: int, current_parts: list, system_text: str, use_functions: bool = True, user_name: str = "User", **kwargs) -> Optional[str]:
    """Main entry: handle Gemini generateContent with function calling, concurrent execution, and proper memory/web_search support."""
    model = await get_gemini_model(cid)
    history = await get_recent_history(cid, CONTEXT_SIZE)

    processed_parts = await _process_parts_for_api(current_parts)

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
            # Send appropriate chat actions concurrently? Use typing for memory/search, etc.
            for fc in function_calls:
                fname = fc["name"]
                if fname in ("save_memory", "load_memory", "web_search"):
                    await send_chat_action(cid, "typing")
                elif fname == "create_pdf":
                    await send_chat_action(cid, "upload_document")
                elif fname == "generate_image":
                    await send_chat_action(cid, "upload_photo")
                
            final_text = await _send_function_response(cid, model, body, function_calls, user_name=user_name)
            if final_text:
                # Extract any sources that may have been embedded by web_search follow-up
                # final_text may already contain formatted_sources if web_search was used
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
    try:
        from api_keys import get_cooldown_stats
        stats = get_cooldown_stats()
        logger.warning(f"handle_gemini failed cid={cid} err={friendly_err[:200]} stats={stats}")
    except Exception:
        pass
    if friendly_err.strip().startswith(("⏳", "⚠️", "❌")):
        error_msg = friendly_err
    else:
        error_msg = f"❌ {friendly_err}"
    await save_message(cid, "model", error_msg)
    await send_message(cid, error_msg)
    return None

