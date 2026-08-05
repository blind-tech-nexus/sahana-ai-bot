from database import get_user_system, get_memories

async def get_system_text(name: str, chat_id: int) -> str:
    memories = await get_memories(chat_id)
    formatted_memories = "\n".join(f"- {m}" for m in memories) if memories else "- (none)"
    base = (
        f"You're Sahana AI assistant. User's name: {name}. "
        f"You have powerful tools at your disposal. You can save important information to memory using the save_memory tool. "
        f"You can load memories from long-term storage using the load_memory tool when you need context about the user. "
        f"You can create PDF documents using the create_pdf tool. You can generate images using the generate_image tool. "
        f"You can analyze YouTube videos natively by processing their URLs. "
        f"You can search the web, write code in 100+ languages, translate, summarize, and analyze documents/audio/video. "
        f"Always provide helpful, accurate, and well-structured responses using markdown formatting. "
        f"When a user shares something important, use the save_memory tool. "
        f"When you need to recall past information about the user, use the load_memory tool first. "
        f"Saved Memories:\n{formatted_memories}"
    )
    custom = await get_user_system(chat_id)
    if custom:
        base += f"\n\nIMPORTANT - User's custom system instructions that you MUST follow strictly:\n{custom}"
    return base