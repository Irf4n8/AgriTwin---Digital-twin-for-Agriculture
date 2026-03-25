import logging
import asyncio
import requests
import pandas as pd
import os
import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters
import json

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# ⚠️ REPLACE WITH YOUR ACTUAL BOT TOKEN & CHAT ID
BOT_TOKEN = "8185619661:AAFW7_v86iVn8CnKFp2rmFQ8vTdLykH4NcU" 
# You can find your Chat ID by messaging @userinfobot
TARGET_CHAT_ID = "949374838" 

API_URL = "http://127.0.0.1:8000"

# ==========================================
# 📝 LOGGING
# ==========================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==========================================
# 🌐 TRANSLATIONS & SETTINGS
# ==========================================
PREFS_FILE = "bot_prefs.json"

def load_prefs():
    try:
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_prefs(prefs):
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f)

USER_LANGS = load_prefs()

TRANSLATIONS = {
    'en': {
        'welcome': "🌾 *Welcome to Agri-Twin Bot!* 🌾\n\nHello {name}!\nI am your digital farm assistant. I will monitor your sensors and alert you if anything goes wrong.\n\nType /help to see all available commands.",
        'backend_error': "⚠️ Could not connect to Agri-Twin Backend.",
        'status_msg': "📊 *Live Farm Status*\n━━━━━━━━━━━━━━━━━━\n🌡 *Temp:* {temp:.1f}°C\n💧 *Humidity:* {hum:.1f}%\n🌱 *Soil Moisture:* {soil:.1f}%\n🌧 *Rainfall:* {rain:.1f} mm\n🧪 *N-P-K:* {n}-{p}-{k}\n━━━━━━━━━━━━━━━━━━\n✅ System Online",
        'status_error': "❌ Error fetching status. Is the backend running?",
        'alert': "🚨 *CRITICAL ALERT* 🚨\n\n{msg}\n\nCheck the dashboard immediately!",
        'lang_prompt': "Please select your preferred language:",
        'lang_changed': "✅ Language set to English.",
        'help': "🛠 *Available Commands*\n\n/start - Register and start the bot\n/tasks - View upcoming farm tasks\n/weather - View current weather forecast\n/soil - View soil sensor readings\n/profit - View your latest profit overview\n/language - Change bot language\n/help - Show this message\n\n*To complete a task:* Type `DONE <task_name>` (e.g., `DONE Irrigation`)",
        'tasks_title': "📋 *Upcoming Tasks*\n━━━━━━━━━━━━━━━━━━\n",
        'tasks_empty': "No pending tasks right now! 🎉",
        'task_done': "✅ Task '{task_name}' marked as completed.",
        'task_not_found': "❌ Could not find an active task matching '{task_name}'.",
        'weather_msg': "🌦 *Weather Update*\n━━━━━━━━━━━━━━━━━━\n🌡 *Temperature:* {temp}°C\n💧 *Humidity:* {hum}%\n☁️ *Condition:* {cond}\n🌧 *Rain Probability:* {rain_prob}%\n━━━━━━━━━━━━━━━━━━\nDrive safe out there!",
        'profit_msg': "💰 *Farm Profit Overview*\n━━━━━━━━━━━━━━━━━━\n💸 *Total Investment:* ₹{cost}\n📈 *Estimated Revenue:* ₹{rev}\n💰 *Expected Profit:* ₹{prof}\n━━━━━━━━━━━━━━━━━━\n{status_emoji}"
    },
    'ta': {
        'welcome': "🌾 *Agri-Twin Bot-க்கு வரவேற்கிறோம்!* 🌾\n\nவணக்கம் {name}!\nநான் உங்கள் டிஜிட்டல் பண்ணை உதவியாளர். நான் உங்கள் சென்சார்களைக் கண்காணித்து, ஏதேனும் தவறாக நடந்தால் உங்களை எச்சரிப்பேன்.\n\nகிடைக்கும் கட்டளைகளைக் காண /help என தட்டச்சு செய்யவும்.",
        'backend_error': "⚠️ Agri-Twin பின்தளத்துடன் இணைக்க முடியவில்லை.",
        'status_msg': "📊 *நேரடி பண்ணை நிலை*\n━━━━━━━━━━━━━━━━━━\n🌡 *வெப்பநிலை:* {temp:.1f}°C\n💧 *ஈரப்பதம்:* {hum:.1f}%\n🌱 *மண் ஈரப்பதம்:* {soil:.1f}%\n🌧 *மழையளவு:* {rain:.1f} மிமீ\n🧪 *N-P-K:* {n}-{p}-{k}\n━━━━━━━━━━━━━━━━━━\n✅ கணினி ஆன்லைனில் உள்ளது",
        'status_error': "❌ நிலையைப் பெறுவதில் பிழை. பின்தளம் இயங்குகிறதா?",
        'alert': "🚨 *முக்கிய எச்சரிக்கை* 🚨\n\n{msg}\n\nஉடனடியாக டாஷ்போர்டை சரிபார்க்கவும்!",
        'lang_prompt': "உங்கள் விருப்பமான மொழியைத் தேர்ந்தெடுக்கவும்:",
        'lang_changed': "✅ மொழி தமிழுக்கு மாற்றப்பட்டது.",
        'help': "🛠 *கிடைக்கும் கட்டளைகள்*\n\n/start - போட்டைத் தொடங்கவும்\n/tasks - வரவிருக்கும் பண்ணை பணிகளைக் காண்க\n/weather - தற்போதைய வானிலை முன்னறிவிப்பைக் காண்க\n/soil - மண் சென்சார் அளவீடுகளைக் காண்க\n/profit - உங்கள் சமீபத்திய லாப மேலோட்டத்தைக் காண்க\n/language - மொழியை மாற்றவும்\n/help - இந்த செய்தியைக் காட்டவும்\n\n*பணியை முடிக்க:* `DONE <பணியின்_பெயர்>` என தட்டச்சு செய்யவும்",
        'tasks_title': "📋 *வரவிருக்கும் பணிகள்*\n━━━━━━━━━━━━━━━━━━\n",
        'tasks_empty': "தற்போதைக்கு நிலுவையில் உள்ள பணிகள் இல்லை! 🎉",
        'task_done': "✅ பணி '{task_name}' முடிந்தது என குறிக்கப்பட்டது.",
        'task_not_found': "❌ '{task_name}' உடன் பொருந்தக்கூடிய செயலில் உள்ள பணியைக் கண்டறிய முடியவில்லை.",
        'weather_msg': "🌦 *வானிலை அப்டேட்*\n━━━━━━━━━━━━━━━━━━\n🌡 *வெப்பநிலை:* {temp}°C\n💧 *ஈரப்பதம்:* {hum}%\n☁️ *நிலை:* {cond}\n🌧 *மழை வாய்ப்பு:* {rain_prob}%\n━━━━━━━━━━━━━━━━━━",
        'profit_msg': "💰 *பண்ணை லாப மேலோட்டம்*\n━━━━━━━━━━━━━━━━━━\n💸 *மொத்த முதலீடு:* ₹{cost}\n📈 *மதிப்பிடப்பட்ட வருவாய்:* ₹{rev}\n💰 *எதிர்பார்க்கப்படும் லாபம்:* ₹{prof}\n━━━━━━━━━━━━━━━━━━\n{status_emoji}"
    }
}

def get_text(chat_id, key):
    lang = USER_LANGS.get(str(chat_id), 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, "")

# ==========================================
# 🤖 BOT COMMANDS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    # Register with backend
    try:
        requests.post(f"{API_URL}/api/bot/register", json={
            "telegram_id": str(chat_id),
            "name": user.first_name
        })
    except Exception as e:
        logging.error(f"Failed to register user: {e}")

    text = get_text(chat_id, 'welcome').format(name=user.first_name)
    await context.bot.send_message(chat_id=chat_id, text=text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = get_text(chat_id, 'help')
    await context.bot.send_message(chat_id=chat_id, text=text, parse_mode='Markdown')

async def fetch_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        res = requests.get(f"{API_URL}/api/tasks?telegram_id={chat_id}")
        data = res.json()
        if data["status"] == "success" and data["data"]:
            msg = get_text(chat_id, 'tasks_title')
            for task in data["data"]:
                due = task['due_time'].split()[0] if task['due_time'] else "TBD"
                msg += f"• *{task['task_name']}* ({task['field_zone']}) - {due}\n"
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.message.reply_text(get_text(chat_id, 'tasks_empty'))
    except Exception as e:
        logging.error(e)
        await update.message.reply_text(get_text(chat_id, 'backend_error'))

async def fetch_weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        res = requests.get(f"{API_URL}/api/weather/current")
        data = res.json()
        msg = get_text(chat_id, 'weather_msg').format(
            temp=data.get('temp', '--'),
            hum=data.get('humidity', '--'),
            cond=data.get('condition', '--'),
            rain_prob=data.get('rain_prob', '--')
        )
        await update.message.reply_text(msg, parse_mode='Markdown')
    except Exception as e:
        logging.error(e)
        await update.message.reply_text(get_text(chat_id, 'backend_error'))

async def fetch_profit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        res = requests.get(f"{API_URL}/api/profit/summary")
        data = res.json()
        if data["status"] == "success":
            prof = data["data"].get("net_profit", 0)
            status_emoji = "🎉 Great Job!" if prof > 0 else "⚠️ Requires attention"
            msg = get_text(chat_id, 'profit_msg').format(
                cost=data["data"].get("total_cost", 0),
                rev=data["data"].get("estimated_revenue", 0),
                prof=prof,
                status_emoji=status_emoji
            )
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.message.reply_text(get_text(chat_id, 'backend_error'))
    except Exception as e:
        logging.error(e)
        await update.message.reply_text(get_text(chat_id, 'backend_error'))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    
    if text.upper().startswith("DONE"):
        task_name = text[4:].strip().lower()
        if not task_name:
            return

        try:
            # Find task ID
            res = requests.get(f"{API_URL}/api/tasks?telegram_id={chat_id}")
            data = res.json()
            matched_task = None
            if data["status"] == "success":
                for task in data["data"]:
                    if task_name in task["task_name"].lower():
                        matched_task = task
                        break
            
            if matched_task:
                requests.post(f"{API_URL}/api/tasks/{matched_task['id']}/complete")
                msg = get_text(chat_id, 'task_done').format(task_name=matched_task['task_name'])
                await update.message.reply_text(msg)
            else:
                msg = get_text(chat_id, 'task_not_found').format(task_name=task_name)
                await update.message.reply_text(msg)
        except Exception as e:
            logging.error(e)
            await update.message.reply_text(get_text(chat_id, 'backend_error'))


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    keyboard = [
        [
            InlineKeyboardButton("English 🇬🇧", callback_data='lang_en'),
            InlineKeyboardButton("தமிழ் 🇮🇳", callback_data='lang_ta')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = get_text(chat_id, 'lang_prompt')
    await update.message.reply_text(text, reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    if query.data == 'lang_en':
        USER_LANGS[str(chat_id)] = 'en'
    elif query.data == 'lang_ta':
        USER_LANGS[str(chat_id)] = 'ta'
        
    save_prefs(USER_LANGS)
    
    text = get_text(chat_id, 'lang_changed')
    await query.edit_message_text(text=text)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        # Fetch from local API
        conn = requests.get(f"{API_URL}/latest")
        if conn.status_code != 200:
            await update.message.reply_text(get_text(chat_id, 'backend_error'))
            return

        data = conn.json()
        
        # Format Message
        msg = get_text(chat_id, 'status_msg').format(
            temp=data.get('temperature', 0),
            hum=data.get('humidity', 0),
            soil=data.get('soil_moisture', 0),
            rain=data.get('rainfall', 0),
            n=data.get('N'),
            p=data.get('P'),
            k=data.get('K')
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logging.error(e)
        await update.message.reply_text(get_text(chat_id, 'status_error'))

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    """Background task to check for critical conditions and send reminders"""
    try:
        # Check Irrigation Status
        res = requests.get(f"{API_URL}/irrigation-alert")
        if res.status_code == 200:
            data = res.json()
            # If critical (you can adjust this logic based on your backend response)
            if "immediately" in data.get("alert_message", "").lower():
                msg = get_text(TARGET_CHAT_ID, 'alert').format(msg=data['alert_message'])
                await context.bot.send_message(
                    chat_id=TARGET_CHAT_ID,
                    text=msg,
                    parse_mode='Markdown'
                )
                
        # Send fake task reminder for demo purposes if there are pending tasks
        res_tasks = requests.get(f"{API_URL}/api/tasks?telegram_id={TARGET_CHAT_ID}")
        if res_tasks.status_code == 200:
            tasks_data = res_tasks.json()
            if tasks_data["status"] == "success" and tasks_data["data"]:
                # Just take the first one and pretend it's due
                task = tasks_data["data"][0]
                reminder_msg = f"🔔 *Farm Task Reminder*\n\nTask: {task['task_name']}\nField: {task['field_zone']}\nTime: Today\n\nPlease complete the task and reply `DONE {task['task_name']}`"
                # Send this exactly once per task to avoid spam? For demo, we might spam it, but let's avoid it by only sending if we haven't sent recently (simplification: don't actually spam every 60s, maybe 1/100 chance for demo to look natural).
                import random
                if random.random() < 0.05:
                    await context.bot.send_message(
                        chat_id=TARGET_CHAT_ID,
                        text=reminder_msg,
                        parse_mode='Markdown'
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
    application.add_handler(CommandHandler('status', status)) # Legacy but keep it mapped to soil
    application.add_handler(CommandHandler('soil', status))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('tasks', fetch_tasks))
    application.add_handler(CommandHandler('weather', fetch_weather))
    application.add_handler(CommandHandler('profit', fetch_profit))
    application.add_handler(CommandHandler('language', language_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Background Job (runs every 60 seconds)
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(alert_job, interval=60, first=10)
    
    print("[INFO] Agri-Twin Bot is running...")
    application.run_polling()
