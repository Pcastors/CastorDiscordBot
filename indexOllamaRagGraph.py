import os
import discord
from dotenv import load_dotenv
from RagSystem import queryQuestion
load_dotenv()

DISCORD_BOT_ID = os.getenv('DISCORDBOT_ID')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DISCORD_CHANNEL = os.getenv('DISCORD_CHANNEL')
LLM_MODEL = os.getenv('LLM_MODEL')
CHROMA_PATH = os.getenv('CHROMA_PATH')
DATA_PATH  = os.getenv('DATA_PATH')
RAG_PROMPT_TEMPLATE = os.getenv('RAG_PROMPT_TEMPLATE')


intents = discord.Intents.default()
intents.guilds = True
intents.guild_messages = True
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print("robotCastor is Ready and online")

@client.event
async def on_message(message):
    if message.author.bot:
        return
    if str(message.channel.id) not in DISCORD_CHANNEL:
        return
    if message.content.startswith('/') or message.content.startswith('!'):
        return
    if f'<@{DISCORD_BOT_ID}>' not in message.content:
        return

    await message.channel.typing()
    #conversation_log = await get_history_log(message)
    #await call_llm(conversation_log, message)
    await call_llm2(message)


async def call_llm2(message):
    try:
        print("reponse recu")
        await manage_response_llm2(message)
    except Exception as e:
        print('Error:', e) 
 
async def manage_response_llm2(message):
    await message.channel.typing()
    messageCleaned = message.content.replace(f"<@{DISCORD_BOT_ID}>", "")
    result = queryQuestion(messageCleaned)
    await message.channel.send(result)


client.run(DISCORD_TOKEN)
