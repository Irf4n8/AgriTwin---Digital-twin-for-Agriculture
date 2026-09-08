# AgriTwin — A Sensor-Free Digital Twin System for Smart Farming

📺 **Demo Video:** https://youtu.be/MBUSTHikvj8
🔗 **Repository:** https://github.com/Irf4n8/AgriTwin---Digital-twin-for-Agriculture

---

## Overview

AgriTwin is a **digital twin platform for agriculture** that creates a virtual representation of farmland using agricultural datasets, real-time weather APIs, and farmer inputs — **without relying on physical IoT sensors**. Instead of expensive sensor hardware, the system uses public agricultural datasets combined with live weather and market-price data to simulate real farm conditions, predict outcomes, and recommend optimal farming strategies.

This project began as a final-year engineering project, and the underlying research was later developed into a paper published in **MSW Management — Multidisciplinary Scientific Work and Management Journal** (Elsevier-indexed, ISSN 1053-7899, Vol. 36, Issue 1, 2026).

---

## Why "Sensor-Free"?

Most smart farming systems depend on physical IoT sensor networks to monitor soil moisture, temperature, and crop conditions — which is expensive to install, maintain, and often impractical for small-scale farmers in developing regions. AgriTwin takes a different approach: it combines **existing agricultural datasets**, **real-time weather APIs**, and **live market-price data** to build an accurate virtual model of the farm, cutting hardware costs while still delivering meaningful, data-driven insights.

---

## Key Features

- **📊 Real-Time Dashboard** — A farm overview showing live environmental readings (temperature, humidity, rainfall), soil health metrics (N-P-K levels, pH), and a 5-day weather forecast.
- **🌱 AI Crop Recommendation** — Suggests the best-fit crop for current soil and weather conditions using a trained classification model.
- **📈 Yield Prediction** — Estimates expected crop yield (in tons/hectare) with a best-case/worst-case range and a confidence score, based on a Random Forest regression model.
- **💹 Market Insights & Live Prices** — Tracks daily vegetable/commodity prices scraped from public market data sources, with price trend charts and today's biggest gainers/fallers.
- **🗺️ Satellite Soil Map** — An interactive Leaflet-powered map where clicking any location fetches real-time surface temperature, sub-surface temperature, soil type, and soil moisture for that spot.
- **🧪 What-If Simulation Engine** — Lets users adjust irrigation, temperature, and fertilizer levels to simulate hypothetical farming scenarios and see the projected impact on yield, cost, and crop stress — without touching real crops.
- **💰 Farm Profit Calculator** — Calculates expected profit/loss based on input costs (seed, fertilizer, pesticide, labor, machinery) against expected yield and market price, with cost-breakdown charts and smart advisory tips.
- **🤖 AI Assistant Chatbot** — An in-dashboard assistant that answers questions about soil health, recommendations, and farming strategies.
- **📲 Telegram Bot Integration** — Farmers can check live farm status, get weather updates, receive critical alerts, and set their preferred language (English / Tamil) directly through Telegram.
- **🌐 Multi-language Support** — Telegram bot supports English and Tamil.

---

## How the "AI" and Recommendation Engine Actually Works

To be transparent about what's under the hood: AgriTwin's intelligence comes from two complementary layers, not a single deep-learning black box.

1. **Rule-Based Evaluation** — A `DecisionEngine` loads a set of configurable rules (`decision_rules.json`) and evaluates farm telemetry/context against them (e.g., if soil nitrogen is low → recommend a fertilizer action). This gives predictable, explainable recommendations.
2. **AI-Driven What-If Optimization** — On top of the rules, a simulation engine (`AgriSimulationEngine`) runs hypothetical scenarios (e.g., "what if irrigation increases by 20%?") and measures the projected yield and cost delta. If a scenario shows a meaningful yield gain relative to its cost, it's surfaced as an AI-optimized recommendation.
3. **Machine Learning Models** — Separate trained models handle the numeric predictions:
   - **Random Forest Classifier** → Crop recommendation
   - **Random Forest Regressor** → Yield prediction (with uncertainty estimation via variance across individual trees)
   - **Random Forest Regressor** → Market/commodity price prediction
   - **Linear Regression / ARIMA** → explored for yield estimation and time-series market trend analysis

So "AI" here refers to a combination of **rule-based logic + simulation-based optimization + classical ML models (Random Forest, Linear Regression, ARIMA)** — not deep learning or neural networks.

---

## Tech Stack

**Backend:**
- Python
- FastAPI (backend server & REST APIs)
- scikit-learn (Random Forest Classifier/Regressor, Linear Regression)
- pandas / numpy (data processing, feature engineering)
- joblib (model persistence)
- SQLite (`agritwin.db`) for structured data storage
- BeautifulSoup + Requests (web scraping live vegetable/commodity prices)
- python-telegram-bot (Telegram bot integration & alert polling)

**Frontend:**
- HTML, CSS, JavaScript (dashboard, forms, charts)
- Leaflet.js (interactive satellite soil map)
- Chart-based visualizations for market trends, environmental trends, and soil nutrient trends

**Data Sources:**
- Public soil health & nutrient datasets
- Weather APIs (temperature, humidity, rainfall)
- Market price data (scraped from public agricultural market sources)
- Farmer-provided inputs (crop, area, cost estimates)

---

## System Architecture

The system follows a layered architecture:

1. **Data Acquisition Layer** — Collects soil datasets, crop datasets, and real-time weather/market data via APIs.
2. **Data Preprocessing Layer** — Cleans, normalizes, and transforms raw data before analysis.
3. **Machine Learning & Analytics Layer** — Runs classification/regression models to generate crop recommendations, yield estimates, and market predictions.
4. **Digital Twin Simulation Layer** — Simulates farming scenarios (irrigation, fertilizer, crop choice) using the outputs of the ML layer.
5. **Visualization & UI Layer** — Presents everything through an interactive dashboard with charts, maps, and a chatbot assistant.

```
Datasets + Weather/Market APIs
        ↓
  FastAPI Backend Server
        ↓
  ML Models (Crop Recommendation, Yield Prediction, Market Analysis)
        ↓
  Real-Time Dashboard, Simulation Module, Telegram Bot
```

---

## Project Report & Publication

This project began as an academic final-year submission and was later developed into a published research paper:

> **AgriTwin: A Sensor-Free Digital Twin System for Smart Farming Using APIs and Geo-Maps**
> *MSW Management – Multidisciplinary Scientific Work and Management Journal*, Vol. 36, Issue 1, 2026 (Elsevier-indexed)

The full project report — covering literature review, system design, dataset engineering, model evaluation, and results — is available in this repository.

---

## Limitations & Future Work

- Currently relies on the availability and quality of public datasets; incomplete or inconsistent regional data can affect prediction accuracy.
- Doesn't yet account for sudden, unpredictable events like extreme weather or pest outbreaks.
- Planned improvements include: optional real IoT sensor integration for hybrid accuracy, deep learning–based crop disease detection from leaf images, and satellite/remote-sensing data integration for large-scale monitoring.

---

## Motivation

AgriTwin was built as an attempt to solve a real, local problem — accessible smart farming for small-scale farmers — using practical, cost-effective technology rather than expensive sensor infrastructure.
