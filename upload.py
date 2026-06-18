from typing import Optional
from config import SUPPORTED_MIME_TYPES, CODE_EXTENSIONS


GEMINI_SUPPORTED_MIMES = {
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/gif",
    "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/opus", "audio/flac", "audio/aac", "audio/webm", "audio/x-wav", "audio/m4a", "audio/x-m4a",
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska", "video/x-msvideo", "video/3gpp",
    "application/pdf",
    "text/plain", "text/html", "text/css", "text/javascript", "text/csv", "text/xml", "application/json", "application/xml", "text/markdown",
}


def is_gemini_supported_mime(mime: str) -> bool:
    mime = (mime or "").strip().lower()
    if mime in GEMINI_SUPPORTED_MIMES:
        return True
    if mime.startswith("text/"):
        return True
    return False


def detect_mime_type(file_path: str, provided_mime: Optional[str] = None) -> str:
    if provided_mime and provided_mime in SUPPORTED_MIME_TYPES:
        return provided_mime
    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    ext_to_mime = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'ppt': 'application/vnd.ms-powerpoint',
        'txt': 'text/plain',
        'csv': 'text/csv',
        'html': 'text/html',
        'css': 'text/css',
        'js': 'text/javascript',
        'json': 'application/json',
        'xml': 'application/xml',
        'py': 'text/x-python',
        'java': 'text/x-java-source',
        'c': 'text/x-c',
        'cpp': 'text/x-c++',
        'cs': 'text/x-csharp',
        'go': 'text/x-go',
        'rs': 'text/x-rust',
        'rb': 'text/x-ruby',
        'php': 'text/x-php',
        'swift': 'text/x-swift',
        'kt': 'text/x-kotlin',
        'scala': 'text/x-scala',
        'sh': 'text/x-shellscript',
        'sql': 'text/x-sql',
        'yaml': 'text/x-yaml',
        'yml': 'text/x-yaml',
        'toml': 'text/x-toml',
        'md': 'text/markdown',
        'ts': 'text/x-typescript',
        'tsx': 'text/x-typescript',
        'jsx': 'text/javascript',
        'lua': 'text/x-lua',
        'pl': 'text/x-perl',
        'r': 'text/x-r',
        'dart': 'text/x-dart',
        'mp3': 'audio/mpeg',
        'm4a': 'audio/mp4',
        'ogg': 'audio/ogg',
        'wav': 'audio/wav',
        'webm': 'audio/webm',
        'flac': 'audio/flac',
        'aac': 'audio/aac',
        'mp4': 'video/mp4',
        'mov': 'video/quicktime',
        'avi': 'video/x-msvideo',
        'mkv': 'video/x-matroska',
        '3gp': 'video/3gpp',
    }
    if ext in ext_to_mime:
        return ext_to_mime[ext]
    if ext in CODE_EXTENSIONS:
        return 'text/plain'
    if provided_mime:
        return provided_mime
    return 'application/octet-stream'


def get_display_name(file_path: str, file_name: Optional[str] = None) -> str:
    if file_name:
        return file_name
    if '.' in file_path:
        return file_path.rsplit('/', 1)[-1]
    return 'uploaded_file'
