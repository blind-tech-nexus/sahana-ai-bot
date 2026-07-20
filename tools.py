from __future__ import annotations

from collections import Counter
from io import BytesIO
import re
import json
from typing import Optional, List, Dict, Any
import asyncio

from api import call_gemini_raw, get_gemini_model
from languages import LANGUAGES
from message import send_document_bytes, send_message, send_photo, send_chat_action
from settings import btn, ikb, tools_keyboard, advanced_tools_keyboard
from texttopdf import execute_text_to_pdf
from config import SHARE_TEXT

TOOL_CLOSE = ikb([[btn("❌ Close", "tools_close")]])
TOOL_CANCEL = ikb([[btn("❌ Cancel", "tools_cancel")]])
TOOL_BACK = ikb([[btn("🔙 Back to Tools", "open_tools")]])
MAX_TOOL_TEXT_FILE_BYTES = 30 * 1024

_LANGUAGE_BY_CODE = {code.lower(): name for name, code in LANGUAGES}
_LANGUAGE_BY_NAME = {name.lower(): code for name, code in LANGUAGES}


def open_tools_text() -> str:
    return (
        "🧰 <b>Welcome to Sahana AI Tools Hub!</b>\n\n"
        "✨ <i>Supercharge your productivity with our advanced AI-powered tools.</i>\n\n"
        "<b>📌 Core Tools:</b>\n"
        "• Text Refiner - Polish your writing\n"
        "• Text Translator - Translate to any language\n"
        "• Text Analyzer - Get detailed insights\n"
        "• PDF Creator - Generate documents\n"
        "• Audio Transcriber - Convert speech to text\n\n"
        "<b>🚀 Advanced AI Tools:</b>\n"
        "• Code Generator - Write code instantly\n"
        "• Content Summarizer - Condense long texts\n"
        "• Email Writer - Craft professional emails\n"
        "• Social Media Post - Create engaging posts\n"
        "• Study Notes - Generate study materials\n"
        "• Recipe Creator - Discover new recipes\n"
        "• Fitness Plan - Build workout routines\n"
        "• Travel Planner - Plan your trips\n"
        "• Business Idea Generator - Spark innovation\n"
        "• Story Writer - Create compelling narratives\n\n"
        "<i>Select a tool below to get started!</i>"
    )


def open_advanced_tools_text() -> str:
    return (
        "🚀 <b>Advanced AI Tools Collection</b>\n\n"
        "<i>Professional-grade tools powered by cutting-edge AI.</i>\n\n"
        "Choose your tool:"
    )


def _safe_tool_file_name(base: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", base).strip("_") or "tool_output.txt"


async def send_tool_long_text(cid: int, text: str, filename: str, caption: str, reply_markup=None) -> None:
    if reply_markup is None:
        reply_markup = TOOL_CLOSE
    
    if len(text) <= 4000:
        await send_message(cid, text, reply_markup=reply_markup, parse_mode="HTML")
        return
    
    await send_document_bytes(
        cid,
        text.encode("utf-8"),
        _safe_tool_file_name(filename),
        caption,
        mime_type="text/plain",
    )
    await send_message(cid, "📄 <i>Full output sent as document.</i>", parse_mode="HTML", reply_markup=reply_markup)


def parse_text_document_bytes(file_bytes: bytes, limit_bytes: Optional[int]) -> tuple[Optional[str], Optional[str]]:
    if limit_bytes is not None and len(file_bytes) > limit_bytes:
        return None, f"❌ File too large. Limit is {limit_bytes // 1024} KB."
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("utf-8-sig")
        except Exception:
            return None, "❌ Could not decode .txt file. Please upload UTF-8 text."
    text = text.strip()
    if not text:
        return None, "❌ Empty text received."
    return text, None


async def run_text_refiner(cid: int, text: str) -> None:
    system = (
        "You are an expert text editor and writing coach. "
        "Refine the following text by enhancing grammar, punctuation, clarity, and flow while preserving the original meaning and tone. "
        "Make it more professional and engaging without changing the core message. "
        "Output ONLY the refined text, no explanations."
    )
    prompt = f"Text to refine:\n\n{text}"
    
    await send_chat_action(cid, "typing")
    refined = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not refined:
        await send_message(cid, "❌ Failed to refine the text. Please try again.", reply_markup=TOOL_CLOSE)
        return
    
    result_text = f"✨ <b>Refined Text</b>\n\n{refined}"
    await send_tool_long_text(cid, result_text, "refined_text.txt", "✅ Your text has been professionally refined.")


async def run_text_translator(cid: int, text: str, lang_code: str, lang_name: str) -> None:
    system = (
        f"You are a professional translator. Translate the following text into {lang_name} ({lang_code}) "
        "with perfect grammar, natural phrasing, and cultural appropriateness. "
        "Maintain the original tone and style. Output ONLY the translation."
    )
    prompt = f"Text to translate:\n\n{text}"
    
    await send_chat_action(cid, "typing")
    translated = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not translated:
        await send_message(cid, "❌ Failed to translate text. Please try again.", reply_markup=TOOL_CLOSE)
        return
    
    result_text = f"🌐 <b>Translation to {lang_name}</b>\n\n{translated}"
    await send_tool_long_text(cid, result_text, f"translated_{lang_code}.txt", f"✅ Translated to {lang_name}")


def resolve_language(target: str) -> tuple[Optional[str], Optional[str]]:
    cleaned = target.strip().lower()
    if cleaned in _LANGUAGE_BY_CODE:
        return cleaned, _LANGUAGE_BY_CODE[cleaned]
    if cleaned in _LANGUAGE_BY_NAME:
        code = _LANGUAGE_BY_NAME[cleaned]
        return code, _LANGUAGE_BY_CODE[code.lower()]
    return None, None


async def run_pdf_creator(cid: int, topic: str) -> None:
    await execute_text_to_pdf(cid, topic)
    await send_message(cid, "📄 <i>PDF created successfully!</i>\n\nYou can create another PDF or close this tool.", 
                      parse_mode="HTML", reply_markup=TOOL_CLOSE)


async def run_text_analyzer(cid: int, text: str) -> None:
    char_count = len(text)
    word_matches = re.findall(r"\b\w+\b", text)
    words = len(word_matches)
    paragraphs = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])
    lines = len(text.splitlines()) if text else 0
    repeated = sum(count for _, count in Counter(w.lower() for w in word_matches).items() if count > 1)
    special_chars = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))
    
    # Calculate reading time (avg 200 words per minute)
    reading_time_minutes = max(1, round(words / 200))
    
    # Get top 5 most frequent words
    word_freq = Counter(w.lower() for w in word_matches)
    top_words = word_freq.most_common(5)
    top_words_str = ", ".join(f"{w}({c})" for w, c in top_words) if top_words else "N/A"
    
    report = (
        "📊 <b>Comprehensive Text Analysis</b>\n\n"
        f"📝 <b>Basic Stats:</b>\n"
        f"• Characters: <code>{char_count:,}</code>\n"
        f"• Words: <code>{words:,}</code>\n"
        f"• Paragraphs: <code>{paragraphs}</code>\n"
        f"• Lines: <code>{lines}</code>\n\n"
        f"📈 <b>Advanced Metrics:</b>\n"
        f"• Reading Time: ~<code>{reading_time_minutes} min</code>\n"
        f"• Avg Word Length: <code>{char_count/words:.1f}</code> chars\n"
        f"• Repeated Words: <code>{repeated}</code>\n"
        f"• Special Characters: <code>{special_chars}</code>\n\n"
        f"🔤 <b>Top 5 Words:</b>\n<code>{top_words_str}</code>"
    )
    await send_message(cid, report, parse_mode="HTML", reply_markup=TOOL_CLOSE)


# ========== NEW ADVANCED TOOLS ==========

async def run_code_generator(cid: int, description: str) -> None:
    """Generate code based on user description."""
    system = (
        "You are an expert programmer. Generate clean, efficient, well-commented code based on the user's requirements. "
        "Include error handling and follow best practices. Provide only the code with brief comments, no lengthy explanations."
    )
    prompt = f"Generate code for: {description}\n\nPlease provide production-ready code with comments."
    
    await send_chat_action(cid, "typing")
    code = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not code:
        await send_message(cid, "❌ Failed to generate code. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"💻 <b>Generated Code</b>\n\n<code>{code}</code>"
    await send_tool_long_text(cid, result_text, "generated_code.txt", "✅ Code generated successfully!")


async def run_content_summarizer(cid: int, text: str) -> None:
    """Summarize long content into key points."""
    system = (
        "You are an expert summarizer. Create a concise, comprehensive summary of the following text. "
        "Extract key points, main ideas, and essential information. Use bullet points for clarity. "
        "Maintain accuracy while reducing length by ~70%."
    )
    prompt = f"Summarize this content:\n\n{text}"
    
    await send_chat_action(cid, "typing")
    summary = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not summary:
        await send_message(cid, "❌ Failed to summarize. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"📝 <b>Content Summary</b>\n\n{summary}"
    await send_tool_long_text(cid, result_text, "summary.txt", "✅ Content summarized successfully!")


async def run_email_writer(cid: int, context: str, tone: str = "professional") -> None:
    """Write professional emails based on context."""
    system = (
        f"You are a professional email writer. Craft a well-structured, polite, and effective email. "
        f"Use a {tone} tone. Include appropriate greeting, clear body, and professional closing. "
        "Make it concise yet complete."
    )
    prompt = f"Write an email with this context: {context}\n\nTone: {tone}"
    
    await send_chat_action(cid, "typing")
    email = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not email:
        await send_message(cid, "❌ Failed to write email. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"📧 <b>Your Professional Email</b>\n\n{email}"
    await send_tool_long_text(cid, result_text, "email_draft.txt", "✅ Email drafted successfully!")


async def run_social_media_post(cid: int, topic: str, platform: str = "Instagram") -> None:
    """Create engaging social media posts."""
    system = (
        f"You are a social media expert. Create an engaging, viral-worthy post for {platform}. "
        "Include relevant emojis, hashtags, and a compelling call-to-action. "
        "Optimize for maximum engagement on this platform."
    )
    prompt = f"Create a {platform} post about: {topic}"
    
    await send_chat_action(cid, "typing")
    post = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not post:
        await send_message(cid, "❌ Failed to create post. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"📱 <b>{platform} Post</b>\n\n{post}"
    await send_tool_long_text(cid, result_text, "social_post.txt", f"✅ {platform} post created!")


async def run_study_notes(cid: int, subject: str, level: str = "intermediate") -> None:
    """Generate comprehensive study notes."""
    system = (
        f"You are an expert educator. Create detailed, well-organized study notes on {subject} for {level} level students. "
        "Include key concepts, definitions, examples, and important formulas if applicable. "
        "Use clear headings and bullet points for easy revision."
    )
    prompt = f"Create comprehensive study notes on {subject} for {level} level."
    
    await send_chat_action(cid, "typing")
    notes = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not notes:
        await send_message(cid, "❌ Failed to generate notes. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"📚 <b>Study Notes: {subject}</b>\n\n{notes}"
    await send_tool_long_text(cid, result_text, f"study_notes_{subject.replace(' ', '_')}.txt", "✅ Study notes generated!")


async def run_recipe_creator(cid: int, ingredients: str, diet_type: str = "any") -> None:
    """Create recipes based on available ingredients."""
    system = (
        f"You are a professional chef. Create a delicious recipe using these ingredients: {ingredients}. "
        f"Dietary preference: {diet_type}. Include title, prep time, cook time, servings, ingredients list, "
        "step-by-step instructions, and nutritional tips."
    )
    prompt = f"Create a recipe with: {ingredients}\nDiet type: {diet_type}"
    
    await send_chat_action(cid, "typing")
    recipe = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not recipe:
        await send_message(cid, "❌ Failed to create recipe. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"🍳 <b>Your Custom Recipe</b>\n\n{recipe}"
    await send_tool_long_text(cid, result_text, "recipe.txt", "✅ Recipe created successfully!")


async def run_fitness_plan(cid: int, goal: str, level: str = "beginner", days: int = 7) -> None:
    """Generate personalized fitness plans."""
    system = (
        f"You are a certified fitness trainer. Create a {days}-day fitness plan for a {level} level person "
        f"with the goal: {goal}. Include daily workouts, exercises, sets, reps, rest periods, and safety tips. "
        "Add warm-up and cool-down routines."
    )
    prompt = f"Create a {days}-day fitness plan. Goal: {goal}, Level: {level}"
    
    await send_chat_action(cid, "typing")
    plan = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not plan:
        await send_message(cid, "❌ Failed to create fitness plan. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"💪 <b>Your {days}-Day Fitness Plan</b>\n\n{plan}"
    await send_tool_long_text(cid, result_text, "fitness_plan.txt", "✅ Fitness plan generated!")


async def run_travel_planner(cid: int, destination: str, duration: str, budget: str = "moderate") -> None:
    """Create detailed travel itineraries."""
    system = (
        f"You are a professional travel planner. Create a detailed {duration} itinerary for {destination} "
        f"with a {budget} budget. Include day-by-day activities, attractions, food recommendations, "
        "transportation tips, estimated costs, and packing suggestions."
    )
    prompt = f"Plan a trip to {destination} for {duration}. Budget: {budget}"
    
    await send_chat_action(cid, "typing")
    itinerary = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not itinerary:
        await send_message(cid, "❌ Failed to create travel plan. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"✈️ <b>Travel Plan: {destination}</b>\n\n{itinerary}"
    await send_tool_long_text(cid, result_text, f"travel_{destination.replace(' ', '_')}.txt", "✅ Travel itinerary created!")


async def run_business_idea_generator(cid: int, industry: str, investment: str = "low") -> None:
    """Generate innovative business ideas."""
    system = (
        f"You are a business consultant and entrepreneur. Generate 5 innovative business ideas in the {industry} sector "
        f"with {investment} initial investment. For each idea, include: concept, target market, revenue model, "
        "unique value proposition, and first steps to launch."
    )
    prompt = f"Generate business ideas in {industry} with {investment} investment."
    
    await send_chat_action(cid, "typing")
    ideas = await call_gemini_raw(cid, [{"text": prompt}], system)
    
    if not ideas:
        await send_message(cid, "❌ Failed to generate ideas. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"💡 <b>Business Ideas in {industry}</b>\n\n{ideas}"
    await send_tool_long_text(cid, result_text, "business_ideas.txt", "✅ Business ideas generated!")


async def run_story_writer(cid: int, genre: str, prompt: str, length: str = "short") -> None:
    """Write creative stories based on prompts."""
    system = (
        f"You are a bestselling author. Write a compelling {length} story in the {genre} genre. "
        "Create interesting characters, engaging plot, vivid descriptions, and a satisfying conclusion. "
        f"Story prompt: {prompt}"
    )
    prompt_full = f"Write a {length} {genre} story based on: {prompt}"
    
    await send_chat_action(cid, "typing")
    story = await call_gemini_raw(cid, [{"text": prompt_full}], system)
    
    if not story:
        await send_message(cid, "❌ Failed to write story. Please try again.", reply_markup=TOOL_BACK)
        return
    
    result_text = f"📖 <b>Your {genre} Story</b>\n\n{story}"
    await send_tool_long_text(cid, result_text, "story.txt", "✅ Story written successfully!")


async def open_tools_menu(cid: int) -> None:
    await send_message(cid, open_tools_text(), parse_mode="HTML", reply_markup=tools_keyboard())


async def open_advanced_tools_menu(cid: int) -> None:
    await send_message(cid, open_advanced_tools_text(), parse_mode="HTML", reply_markup=advanced_tools_keyboard())
