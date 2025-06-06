# main.py

import os
import json
import faiss
import numpy as np
import telegram
from telegram.ext import Updater, MessageHandler, Filters
from sentence_transformers import SentenceTransformer
import requests

# --- Config ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TEXT_FOLDER = "texts"  # folder with ethical/religious texts (txt files)
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# --- Load and embed documents ---
documents = []
texts = []
for filename in os.listdir(TEXT_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(TEXT_FOLDER, filename), 'r', encoding='utf-8') as f:
            chunks = f.read().split("\n\n")
            texts.extend(chunks)
            documents.extend([(chunk, filename) for chunk in chunks])

embeddings = EMBEDDING_MODEL.encode([doc[0] for doc in documents], convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# --- Message history storage ---
conversation_history = {}

# --- Telegram bot handler ---
def respond(update, context):
    user_id = str(update.message.chat_id)
    user_input = update.message.text.strip()

    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": "user", "content": user_input})

    query_embedding = EMBEDDING_MODEL.encode([user_input], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=3)
    relevant_chunks = [documents[i][0] for i in I[0]]

    history = conversation_history[user_id][-5:]
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = f"""
شما یک دستیار آموزشی هستید که از قرآن، نهج‌البلاغه و متون اخلاقی برای پاسخ دادن به سوالات دانش‌آموزان استفاده می‌کند. با زبانی ساده و آموزنده پاسخ دهید.

تاریخچه گفتگو:
{history_text}

متن‌های مرجع:
{chr(10).join(relevant_chunks)}

پاسخ به آخرین سوال:
"""

    response = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": TOGETHER_MODEL,
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7
        }
    )

    result = response.json()
    answer = result.get("choices", [{}])[0].get("text", "متاسفم، مشکلی در پاسخ‌دهی پیش آمده.").strip()
    conversation_history[user_id].append({"role": "assistant", "content": answer})
    update.message.reply_text(answer)

# --- Start bot ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
updater = Updater(TELEGRAM_TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(MessageHandler(Filters.text & ~Filters.command, respond))

print("🤖 Bot is running...")
updater.start_polling()
updater.idle()
