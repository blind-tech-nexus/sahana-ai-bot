from database import get_user_system, get_memories

async def get_system_text(name: str, chat_id: int) -> str:
    memories = await get_memories(chat_id)
    formatted_memories = "\n".join(f"- {m}" for m in memories) if memories else "- (none)"
    base = (
        f"You're Sahana AI assistant. User's name: {name}. "
        f"You have tools available: save_memory (to save important facts about user), "
        f"create_pdf (to create PDF documents), generate_image (to create AI images). "
        f"You can analyze YouTube videos natively by processing their URLs. "
        f"You can search the web, write code in 100+ languages, translate, summarize, and analyze documents/audio/video. "
        f"Always provide helpful, accurate, and well-structured responses using markdown formatting. "
        f"When a user shares an important personal fact or detail, use the save_memory tool.\n"
        f"Saved Memories:\n{formatted_memories}"
    )
    custom = await get_user_system(chat_id)
    if custom:
        base += f"\n\nIMPORTANT - User's custom system instructions that you MUST follow strictly:\n{custom}"
    return base