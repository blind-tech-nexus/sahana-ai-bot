import os

BOT_TOKEN = os.environ.get("bot_token")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
POOL_API = "https://sr-pool-api-5bm.pages.dev"
USER_MODEL = "gemini-3-flash-preview"
ADMIN_MODEL = "gemini-3-flash-preview"
ADMINS = [7026190306, 6280547580]
MICROSOFT_TTS_API = "https://multi-functional-api-sujan.vercel.app/tts/Microsoft"
DEFAULT_TTS_VOICE = "en-US-AriaNeural"
REDIS_URL = os.environ.get("REDIS_URL", "")
BOT_USERNAME = os.environ.get("BOT_USERNAME", "meroaiassistantbot_bot")
BOT_MENTION_ALIASES = [a.strip() for a in os.environ.get("BOT_MENTION_ALIASES", "").split(",") if a.strip()]
MAX_HISTORY = 1000
CONTEXT_SIZE = 50

SHARE_TEXT = "🚀 Check out Daily AI Companion — your free, fast & powerful AI companion on Telegram!\n\nhttps://t.me/meroaiassistantbot_bot"

AGENT_PROMPT = """You're an AI agent for a telegram bot built with python. Your task is to return the specified function with parameters as told. You have to understand prompt and return necessary function.
You are Daily AI Companion's routing brain. Analyze the user prompt and return one or multiple python function calls.
Available functions:

saveMemory(userId, memory)
- Use only when any part of the prompt is important to remember forever.
- userId must be returned as raw variable userId (not quoted) because it is auto-passed.
- memory must be a short string with the memory content.

processYoutube(prompt, link)
- Use when the user message contains a YouTube URL and asks to summarize, explain, analyze, extract insights, or transcribe.
- Always extract both a clean prompt and the URL.
- If prompt is missing, use: "Summarize and transcribe this YouTube video".

generateImage(query)
- Use when the user asks to generate/create/draw an image.

texttopdf(prompt)
- Use when the user asks to create/generate a PDF from text/topic.
- Pass the extracted user intent as prompt.

Rules:
- You can return one or two function calls.
- If returning multiple, put each function on a new line. For example, you can return save memory and process youtube functions at the same time.
- No markdown, no backticks, no extra text.
- Keep parameters as strings when possible.
- Never return JSON.
- Never ask follow-up questions in router mode.
- Prioritize accuracy over creativity.
- The agent will only return functions for YouTube link processing, pdf creation, image generation, and to save important memories.
- Reply if simple tasks that don't require these functions - just return the function call for the task.
- If uncertain about special tasks, do not return any function - let the default chat handle it.
User prompt: {user_prompt}"""

TEMPLATE_PROMPTS = [
    "Explain quantum computing simply",
    "Write a Python web scraper",
    "Summarize the latest AI news",
    "Translate 'hello' to 10 languages",
    "Solve: integral of x²·sin(x) dx",
    "Generate a business plan outline",
    "Explain blockchain in 3 sentences",
    "Write a poem about the ocean",
    "Compare React vs Vue vs Angular",
    "Tips for learning a new language",
]

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "ppt",
    "text/plain": "txt",
    "text/csv": "csv",
    "text/html": "html",
    "text/css": "css",
    "text/javascript": "js",
    "application/json": "json",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/x-python": "py",
    "text/x-java-source": "java",
    "text/x-c": "c",
    "text/x-c++": "cpp",
    "text/x-csharp": "cs",
    "text/x-go": "go",
    "text/x-rust": "rs",
    "text/x-ruby": "rb",
    "text/x-php": "php",
    "text/x-swift": "swift",
    "text/x-kotlin": "kt",
    "text/x-scala": "scala",
    "text/x-shellscript": "sh",
    "text/x-sql": "sql",
    "text/x-yaml": "yaml",
    "text/x-toml": "toml",
    "text/markdown": "md",
    "text/x-typescript": "ts",
    "text/x-lua": "lua",
    "text/x-perl": "pl",
    "text/x-r": "r",
    "text/x-dart": "dart",
    "application/x-httpd-php": "php",
    "application/javascript": "js",
    "application/typescript": "ts",
    "application/x-yaml": "yaml",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/x-wav": "wav",
    "audio/flac": "flac",
    "audio/aac": "aac",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/x-matroska": "mkv",
    "video/3gpp": "3gp",
}

CODE_EXTENSIONS = {
    "py", "js", "ts", "java", "c", "cpp", "cs", "go", "rs", "rb", "php",
    "swift", "kt", "scala", "sh", "sql", "yaml", "yml", "toml", "md",
    "html", "css", "json", "xml", "lua", "pl", "r", "dart", "jsx", "tsx",
    "vue", "svelte", "zig", "nim", "ex", "exs", "clj", "hs", "ml", "fs",
    "v", "d", "pas", "bas", "asm", "s", "coffee", "elm", "erl", "groovy",
    "tf", "dockerfile", "makefile", "cmake", "gradle", "bat", "ps1",
    "ini", "cfg", "conf", "env", "gitignore", "editorconfig", "txt",
    "csv", "tsv", "log", "diff", "patch",
}
