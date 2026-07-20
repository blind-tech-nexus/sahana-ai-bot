import urllib.parse
import httpx
from database import save_message
from message import send_message, send_photo
from settings import ikb, btn


async def execute_image(cid: int, query: str, name: str) -> None:
    await send_message(cid, "🎨 Generating image...", reply_markup=ikb([[btn("⏳ Creating...", "noop")]]))
    encoded_prompt = urllib.parse.quote(query)
    image_api_url = f"https://yabes-api.pages.dev/api/ai/image/dalle?prompt={encoded_prompt}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(image_api_url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and "output" in data:
                    await send_photo(cid, data["output"], f"🎨 {query}", reply_markup=ikb([[btn("🔄 Regenerate", f"regen_img:{query[:60]}")]]))
                    await save_message(cid, "user", f"Generate image: {query}")
                    await save_message(cid, "model", f"Generated image for: {query}")
                    return
        await send_message(cid, "❌ Image generation failed. Please try again.")
    except Exception as e:
        await send_message(cid, f"❌ Image generation error: {e}")
