import re
import html as _html

# Allowed Telegram HTML tags that should NOT be escaped — they will be preserved as-is.
# Includes: b, strong, i, em, u, ins, s, strike, del, code, pre, a, span (for tg-spoiler), blockquote, tg-emoji, tg-spoiler
_ALLOWED_HTML_RE = re.compile(
    r"</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|a|span|blockquote|tg-spoiler|tg-emoji)(?:\s+[^>]*?)?>",
    re.IGNORECASE,
)


def escape_html(text: str) -> str:
    # Keep minimal escaping: &, <, >, " — Telegram HTML requires these.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _format_inline(text: str) -> str:
    # text is already escaped except for placeholders (XXHTMLTAG* / XXINLINECODE*)
    # Protect inline code `code` first so markdown inside code is not processed
    # Use placeholders WITHOUT markdown chars (_ * ` [ ] ) to avoid interference
    code_placeholders: list[str] = []

    def _code_repl(m: re.Match) -> str:
        content = m.group(1)  # already escaped
        ph = f"XXINLINECODE{len(code_placeholders)}XX"
        code_placeholders.append(f"<code>{content}</code>")
        return ph

    text = re.sub(r"`([^`\n]+?)`", _code_repl, text)

    # Markdown links: [text](https://url) -> <a href="url">text</a>
    # Handle href escaping; url may already be escaped (since we escape whole line before this)
    # Decode then re-encode to avoid double escaping like &amp;amp;
    def _link_repl(m: re.Match) -> str:
        link_text = m.group(1)
        url = m.group(2).strip()
        # Decode common entities if already escaped, then re-escape once
        url_decoded = url.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
        safe_url = url_decoded.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        return f'<a href="{safe_url}">{link_text}</a>'

    text = re.sub(r"\[([^\]]+?)\]\((https?://[^\s)]+?)\)", _link_repl, text)

    # Bold + Italic: ***text*** -> <b><i>text</i></b>
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    # Bold: **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Underline via __text__ -> <u> (Telegram supports <u>/<ins>). Keep ** for bold, __ for underline per Telegram spec
    text = re.sub(r"__(.+?)__", r"<u>\1</u>", text)
    # Italic via *text* (single star, not part of **)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)
    # Italic via _text_ (single underscore)
    text = re.sub(r"(?<!_)_(?!_)([^_\n]+?)_(?!_)", r"<i>\1</i>", text)
    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    # Spoiler: ||text||
    text = re.sub(r"\|\|(.+?)\|\|", r'<span class="tg-spoiler">\1</span>', text)

    # Restore inline code placeholders
    for i, html_code in enumerate(code_placeholders):
        text = text.replace(f"XXINLINECODE{i}XX", html_code)
    return text


def _render_text_block(block: str) -> str:
    # Preserve allowed HTML tags before escaping: replace them with placeholders
    # Use placeholder without _ * ` to avoid markdown processing interference
    html_tags: list[str] = []

    def _tag_repl(m: re.Match) -> str:
        html_tags.append(m.group(0))
        return f"XXHTMLTAG{len(html_tags)-1}XX"

    # Save allowed tags (case-insensitive) before any escaping
    protected_block = _ALLOWED_HTML_RE.sub(_tag_repl, block)

    lines = protected_block.split("\n")
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not line:
            out.append("")
            continue

        # Heading: # .. ######  (render as bold in Telegram, since no <h1> support)
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            content_raw = heading.group(2).strip()
            # Escape then format inline (placeholders stay)
            escaped = escape_html(content_raw)
            formatted = _format_inline(escaped)
            out.append(f"<b>{formatted}</b>")
            continue

        # Unordered list: - or * + space
        unordered = re.match(r"^\s*[-*]\s+(.*)$", line)
        if unordered:
            content_raw = unordered.group(1).strip()
            escaped = escape_html(content_raw)
            formatted = _format_inline(escaped)
            out.append(f"• {formatted}")
            continue

        # Ordered list: 1. or 1) + space
        ordered = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if ordered:
            idx = ordered.group(1)
            content_raw = ordered.group(2).strip()
            escaped = escape_html(content_raw)
            formatted = _format_inline(escaped)
            out.append(f"{idx}. {formatted}")
            continue

        # Blockquote: > text
        quote = re.match(r"^>\s?(.*)$", line)
        if quote:
            content_raw = quote.group(1)
            escaped = escape_html(content_raw)
            formatted = _format_inline(escaped)
            out.append(f"│ {formatted}")
            continue

        # Normal line
        escaped = escape_html(line)
        formatted = _format_inline(escaped)
        out.append(formatted)

    # Join and then restore allowed HTML tags placeholders globally
    joined = "\n".join(out)
    for i, tag in enumerate(html_tags):
        joined = joined.replace(f"XXHTMLTAG{i}XX", tag)
    return joined


def markdown_to_html(text: str) -> str:
    if not text:
        return ""
    # Extract fenced code blocks ```...``` before any other processing
    # Preserve them as separate parts to avoid markdown processing inside.
    pattern = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)
    parts: list[tuple] = []
    last = 0
    for match in pattern.finditer(text):
        before = text[last:match.start()]
        if before:
            parts.append(("text", before))
        # group(1) = language, group(2) = code content
        lang_raw = (match.group(1) or "").strip()
        code_content = match.group(2).rstrip("\n")
        parts.append(("code", lang_raw, code_content))
        last = match.end()
    tail = text[last:]
    if tail:
        parts.append(("text", tail))
    if not parts:
        return _render_text_block(text)

    rendered: list[str] = []
    for part in parts:
        if part[0] == "code":
            # Sanitize language: only allow alphanum + - _ to prevent attribute injection
            lang_raw = part[1]
            safe_lang = re.sub(r"[^a-zA-Z0-9_-]", "", lang_raw)
            code = escape_html(part[2])
            # Per Telegram best practice (zeroclaw fix): use <pre> without class attribute to avoid injection and ensure rendering
            # If language needed, include as plain text inside? Telegram ignores class anyway, so just use <pre>
            # Use <pre><code> wrapper for compatibility, without class attribute
            if safe_lang:
                # Keep language as comment inside? Simpler to just use <pre> without class
                rendered.append(f"<pre>{code}</pre>")
            else:
                rendered.append(f"<pre>{code}</pre>")
        else:
            rendered.append(_render_text_block(part[1]))
    # Join with newline to separate code and text blocks correctly
    return "\n".join(p for p in rendered if p is not None)
