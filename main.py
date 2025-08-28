import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import httpx
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize bot and dispatcher
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is not set")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Initialize OpenAI client
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b:free")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_URL
)

# Maximum length for Telegram messages
TELEGRAM_MAX_LENGTH = 4096

# In-memory storage for tracking queries (in a production app, use a database)
query_storage = {}

# Cache for free models
free_models_cache: List[str] = []
cache_timestamp: float = 0
CACHE_DURATION = 300  # 5 minutes

async def fetch_free_models() -> List[str]:
    """
    Fetch free models from OpenRouter API.
    """
    global free_models_cache, cache_timestamp
    
    # Check if cache is still valid
    current_time = asyncio.get_event_loop().time()
    if free_models_cache and (current_time - cache_timestamp) < CACHE_DURATION:
        return free_models_cache
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                params={"max_price": 0},
                headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {},
            )
            response.raise_for_status()
            
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            
            # Update cache
            free_models_cache = models
            cache_timestamp = current_time
            
            return models
    except Exception as e:
        logger.error(f"Error fetching free models: {e}")
        return free_models_cache or []  # Return cached models or empty list

def is_valid_model(model: str) -> bool:
    """
    Check if a model is valid (exists in free models).
    """
    # For now, we'll allow any model since we can't easily validate without API call
    # In a production implementation, you would check against the fetched models
    return True

def extract_model_and_query(text: str) -> tuple[Optional[str], str]:
    """
    Extract model and query from text.
    Expected format: "[bot] #model_name query" or "[bot] query"
    """
    # Remove the bot mention if present
    if text.startswith("[bot]"):
        text = text[5:].strip()  # Remove "[bot]" prefix
    elif text.startswith("@inline_llm_bot"):
        text = text[16:].strip()  # Remove "@inline_llm_bot" prefix
    
    # Check for model specification
    model_pattern = r"^#(\S+)\s+(.+)$"
    match = re.match(model_pattern, text)
    
    if match:
        model = match.group(1)
        query = match.group(2)
        return model, query
    else:
        # No model specified, return default model and full text
        return None, text

async def query_llm(model: Optional[str], prompt: str) -> str:
    """
    Query the LLM with the given model and prompt.
    """
    try:
        # Use the specified model or default
        model_name = model if model else DEFAULT_MODEL
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,  # Set to a reasonable value, Telegram will truncate if needed
            temperature=0.7,
            extra_body={"reasoning": {"exclude": True}}
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Hide reasoning if present (simple approach - remove content between <think> tags if they exist)
        if content:
            # Remove any reasoning/thinking patterns
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"(\(.*?reasoning.*?\))", "", content, flags=re.IGNORECASE)
            content = re.sub(r"(\[.*?reasoning.*?\])", "", content, flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            content = re.sub(r"\n\s*\n", "\n\n", content).strip()
            
        return content or "No response from model."
        
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return f"Error: {str(e)}"

@dp.inline_query()
async def inline_query_handler(inline_query: types.InlineQuery):
    """
    Handle inline queries.
    """
    query_text = inline_query.query
    
    if not query_text:
        # Return empty result if no query
        await bot.answer_inline_query(inline_query.id, results=[], cache_time=1)
        return
    
    # Extract model and query
    model, prompt = extract_model_and_query(query_text)
    
    # Validate model if specified
    if model and not is_valid_model(model):
        # Return error result for invalid model
        result = types.InlineQueryResultArticle(
            id="invalid_model",
            title="Invalid Model",
            description=f"Model '{model}' is not available. Use /models to see available models.",
            input_message_content=types.InputTextMessageContent(
                message_text=f"❌ Error: Model '{model}' is not available.\n\nSee available models with /models command."
            )
        )
        await bot.answer_inline_query(inline_query.id, results=[result], cache_time=1, is_personal=True)
        return
    
    if not prompt:
        # Return empty result if no prompt
        await bot.answer_inline_query(inline_query.id, results=[], cache_time=1)
        return
    
    # Store the query data for later retrieval
    query_id = str(uuid.uuid4())[:8]  # Short ID for tracking
    query_storage[query_id] = {
        "model": model,
        "prompt": prompt,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    # Clean up old queries (older than 1 hour)
    current_time = asyncio.get_event_loop().time()
    expired_keys = [key for key, data in query_storage.items() 
                   if current_time - data["timestamp"] > 3600]
    for key in expired_keys:
        del query_storage[key]
    
    # Create inline query result with processing message and a dummy button
    result = types.InlineQueryResultArticle(
        id=query_id,
        title="Generate LLM Response",
        description=f"Generate response for: {prompt[:50]}{'...' if len(prompt) > 50 else ''}",
        input_message_content=types.InputTextMessageContent(
            message_text="⏳ Processing your request...",
        ),
        reply_markup=types.InlineKeyboardMarkup(
            inline_keyboard=[
                [types.InlineKeyboardButton(
                    text="⏳ Processing...", 
                    callback_data="dummy"
                )]
            ]
        )
    )
    
    # Answer the inline query immediately
    await bot.answer_inline_query(
        inline_query.id, 
        results=[result], 
        cache_time=0,  # Don't cache for immediate processing
        is_personal=True
    )

@dp.chosen_inline_result()
async def chosen_inline_result_handler(chosen_inline_result: types.ChosenInlineResult):
    """
    Handle chosen inline results and process the query.
    """
    query_id = chosen_inline_result.result_id
    inline_message_id = chosen_inline_result.inline_message_id
    
    # Check if we have the query data
    if not query_id or query_id not in query_storage:
        logger.error(f"Query not found: {query_id}")
        return
    
    query_data = query_storage[query_id]
    model = query_data["model"]
    prompt = query_data["prompt"]
    
    # Remove the query from storage
    del query_storage[query_id]
    
    try:
        # Edit the message to show processing
        await bot.edit_message_text(
            inline_message_id=inline_message_id,
            text="🔄 Generating response..."
        )
        
        # Query the LLM
        response_text = await query_llm(model, prompt)
        
        # Truncate response to Telegram's maximum length if needed
        if len(response_text) > TELEGRAM_MAX_LENGTH:
            response_text = response_text[:TELEGRAM_MAX_LENGTH-3] + "..."
        
        # Edit the message with the response
        await bot.edit_message_text(
            inline_message_id=inline_message_id,
            text=response_text,
            reply_markup=types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [types.InlineKeyboardButton(
                        text="✅ Completed", 
                        callback_data="dummy"
                    )]
                ]
            )
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        try:
            await bot.edit_message_text(
                inline_message_id=inline_message_id,
                text="❌ Error generating response"
            )
        except Exception as edit_error:
            logger.error(f"Error editing message: {edit_error}")

@dp.callback_query()
async def dummy_callback_handler(callback_query: types.CallbackQuery):
    """
    Handle dummy button callbacks.
    """
    # Just answer with a notification
    await callback_query.answer()

@dp.message(Command("start", "help"))
async def send_welcome(message: types.Message):
    """
    Send welcome message with instructions.
    """
    help_text = (
        "I'm an inline LLM bot! Here's how to use me:\n\n"
        "In any chat, type my username followed by your query:\n"
        "`@inline_llm_bot What is the capital of France?`\n\n"
        "To specify a model, use the #model syntax:\n"
        "`@inline_llm_bot #openai/gpt-3.5-turbo Hello!`\n\n"
        "I'll hide any reasoning/thoughts from the model response.\n\n"
        "Use /models to see available free models."
    )
    await message.answer(help_text, parse_mode="Markdown")

@dp.message(Command("models"))
async def send_models(message: types.Message):
    """
    Send list of available free models.
    """
    try:
        models = await fetch_free_models()
        
        if not models:
            await message.answer("❌ Error fetching models. Please try again later.")
            return
        
        # Create a message with the models
        models_text = "Available free models:\n\n"
        for model in models[:20]:  # Limit to first 20 models
            models_text += f"• {model}\n"
        
        if len(models) > 20:
            models_text += f"\n... and {len(models) - 20} more models."
        
        models_text += "\n\nSee all free models at: https://openrouter.ai/models?max_price=0&order=top-weekly"
        
        # Split into multiple messages if too long
        if len(models_text) > TELEGRAM_MAX_LENGTH:
            chunks = [models_text[i:i+TELEGRAM_MAX_LENGTH] for i in range(0, len(models_text), TELEGRAM_MAX_LENGTH)]
            for chunk in chunks:
                await message.answer(chunk)
        else:
            await message.answer(models_text)
            
    except Exception as e:
        logger.error(f"Error sending models: {e}")
        await message.answer("❌ Error fetching models. Please try again later.")

async def main():
    """
    Main function to start the bot.
    """
    logger.info("Starting bot...")
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")

if __name__ == "__main__":
    asyncio.run(main())
