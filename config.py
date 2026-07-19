import os

BOT_TOKEN = os.environ.get("bot_token")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
POOL_API = "https://sr-pool-api-5bm.pages.dev"

# Models
MODEL_SAHANA_1 = "gemini-2.5-flash"
MODEL_SAHANA_2 = "gemini-3-flash-preview"
MODEL_SAHANA_3 = "gemini-3.5-flash"
DEFAULT_MODEL = "sahana-1"

ADMINS = [7026190306, 6280547580]
DEFAULT_TTS_VOICE = "en-US-AriaNeural"
REDIS_URL = os.environ.get("REDIS_URL", "")

BOT_USERNAME = os.environ.get("BOT_USERNAME", "sahanaraiai_bot")
BOT_MENTION_ALIASES = [a.strip() for a in os.environ.get("BOT_MENTION_ALIASES", "").split(",") if a.strip()]

MAX_HISTORY = 1000
CONTEXT_SIZE = 50

SHARE_TEXT = "🚀 Check out Sahana AI — your free, fast & powerful AI companion on Telegram!\nhttps://t.me/meroaiassistantbot_bot"

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
    "application/pdf": "pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls", "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "ppt", "text/plain": "txt", "text/csv": "csv", "text/html": "html",
    "text/css": "css", "text/javascript": "js", "application/json": "json", "application/xml": "xml",
    "text/xml": "xml", "text/markdown": "md", "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/ogg": "ogg",
    "audio/wav": "wav", "audio/webm": "webm", "audio/flac": "flac", "audio/aac": "aac", "video/mp4": "mp4",
    "video/webm": "webm", "video/quicktime": "mov", "video/x-msvideo": "avi", "video/x-matroska": "mkv", "video/3gpp": "3gp",
}

CODE_EXTENSIONS = {
    "py", "js", "ts", "java", "c", "cpp", "cs", "go", "rs", "rb", "php", "swift", "kt", "scala", "sh", "sql", "yaml", "yml",
    "toml", "md", "html", "css", "json", "xml", "lua", "pl", "r", "dart", "jsx", "tsx", "vue", "svelte", "txt", "csv", "log",
}