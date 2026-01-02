import logging
import asyncio
import requests
import pandas as pd
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# ⚠️ REPLACE WITH YOUR ACTUAL BOT TOKEN & CHAT ID
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
# You can find your Chat ID by messaging @userinfobot
TARGET_CHAT_ID = "YOUR_CHAT_ID_HERE" 

API_URL = "http://127.0.0.1:8000"

# ==========================================
# 📝 LOGGING
# ==========================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==========================================
# 🤖 BOT COMMANDS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"🌾 *Welcome to Agri-Twin Bot!* 🌾\n\nHello {user.first_name}!\nI am your digital farm assistant. I will monitor your sensors and alert you if anything goes wrong.\n\nType /status to see current farm conditions."
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Fetch from local API
        conn = requests.get(f"{API_URL}/latest")
        if conn.status_code != 200:
            await update.message.reply_text("⚠️ Could not connect to Agri-Twin Backend.")
            return

        data = conn.json()
        
        # Format Message
        msg = (
            f"📊 *Live Farm Status*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🌡 *Temp:* {data.get('temperature', 0):.1f}°C\n"
            f"💧 *Humidity:* {data.get('humidity', 0):.1f}%\n"
            f"🌱 *Soil Moisture:* {data.get('soil_moisture', 0):.1f}%\n"
            f"🌧 *Rainfall:* {data.get('rainfall', 0):.1f} mm\n"
            f"🧪 *N-P-K:* {data.get('N')}-{data.get('P')}-{data.get('K')}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"✅ System Online"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logging.error(e)
        await update.message.reply_text("❌ Error fetching status. Is the backend running?")

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    """Background task to check for critical conditions"""
    try:
        # Check Irrigation Status
        res = requests.get(f"{API_URL}/irrigation-alert")
        if res.status_code == 200:
            data = res.json()
            # If critical (you can adjust this logic based on your backend response)
            if "immediately" in data.get("alert_message", "").lower():
                await context.bot.send_message(
                    chat_id=TARGET_CHAT_ID,
                    text=f"🚨 *CRITICAL ALERT* 🚨\n\n{data['alert_message']}\n\nCheck the dashboard immediately!"
                )
    except Exception as e:
        logging.error(f"Alert job failed: {e}")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    if not BOT_TOKEN or "YOUR_BOT_TOKEN" in BOT_TOKEN:
        print("❌ ERROR: Please set a valid BOT_TOKEN in telegram_bot.py")
        exit(1)

    application = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('status', status))
    
    # Background Job (runs every 60 seconds)
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(alert_job, interval=60, first=10)
    
    print("🤖 Agri-Twin Bot is running...")
    application.run_polling()
