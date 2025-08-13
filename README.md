# Inline LLM Bot

A Telegram bot that works in inline mode to query Large Language Models (LLMs) directly from any chat.

## Features

- Works in Telegram's inline mode - no need to switch chats
- Specify different models using `#model_name` syntax
- Uses default model when none specified
- Hides model reasoning/thoughts from responses
- Respects Telegram's maximum message length

## Usage

In any Telegram chat, simply type the bot's username followed by your query:

```
@inline_llm_bot What is the capital of France?
```

To specify a model, use the `#model_name` syntax:

```
@inline_llm_bot #openai/gpt-3.5-turbo Hello!
```

The bot will show a "Generate Response" button. Click this button to generate the LLM response. The message will first show "Processing your request..." and then automatically update with the LLM's response once it's generated.

## Setup

1. Create a Telegram bot using [@BotFather](https://t.me/BotFather)
2. Enable inline mode for your bot in BotFather settings
3. Set the following environment variables in a `.env` file:
   - `BOT_TOKEN`: Your Telegram bot token
   - `API_KEY`: Your OpenAI-compatible API key
   - `API_URL`: The API endpoint URL (optional, defaults to OpenRouter)
   - `DEFAULT_MODEL`: The default model to use (optional)

Example `.env` file:
```env
BOT_TOKEN=your_telegram_bot_token
API_KEY=your_openai_api_key
API_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=openai/gpt-oss-20b:free
```

## Running the Bot

Install dependencies using `uv`:

```bash
uv sync
```

Run the bot:

```bash
uv run bot.py
```

## Dependencies

- Python 3.13+
- aiogram 3.x - Telegram bot framework
- openai - OpenAI API client
- python-dotenv - Environment variable management

All dependencies are managed using `uv` and specified in `pyproject.toml`.
