# 📊 AI Live Stock Intelligence Dashboard

An AI-powered web dashboard that lets you search for **any publicly listed company in the world** — big or small, Indian or global — and instantly get live prices, technical indicators, and a 7-day AI-generated price forecast, all displayed in Indian Rupees (₹).

Built with **Flask**, **yfinance**, **Plotly.js**, and an **LSTM neural network** (Keras/TensorFlow).

---

## ✨ Features

- 🔎 **Global company search** — type a company name (not just a ticker) like "Tata Motors", "Apple", or an obscure small-cap, and the app resolves it to the correct listed ticker automatically.
- ✅ **Validated ticker resolution** — every candidate ticker is checked for real trading history before being used, so results are accurate even for thinly-covered small caps.
- 🚫 **Honest handling of private companies** — if a searched company isn't publicly traded (e.g. an unlisted startup that hasn't IPO'd), the app clearly explains why instead of throwing a generic error.
- 💹 **Live market data in ₹** — automatic currency conversion to INR for any foreign-listed stock (USD, EUR, GBP, JPY, etc.).
- 📈 **Interactive charts** — switch between line and candlestick views, with 20-day SMA overlay, powered by Plotly.js.
- 🤖 **7-day AI price forecast** — an LSTM neural network trained per-ticker predicts the next 7 trading sessions.
- 🎯 **Risk management plan** — auto-generated target price, stop-loss, and risk:reward ratio based on the forecast and ATR (Average True Range).
- 📊 **Technical indicators** — RSI (14), SMA (20), 52-week high/low, day range, and volume.
- 📥 **Export options** — download historical data as CSV, or print/save the dashboard as PDF.
- ⏱️ **Multiple timeframes** — 1M, 6M, 1Y, and 5Y views.
- ⚡ **Caching** — live data cached for 3 minutes and trained models cached for 12 hours to keep the app fast and avoid unnecessary retraining/API calls.

---

## 🛠️ Tech Stack

| Layer            | Technology                          |
|-------------------|--------------------------------------|
| Backend           | Flask (Python)                      |
| Market data       | yfinance / Yahoo Finance             |
| Forecasting model | LSTM (Keras / TensorFlow)            |
| Preprocessing     | scikit-learn (MinMaxScaler)          |
| Charts            | Plotly.js                           |
| Frontend          | HTML5, CSS3, Jinja2 templates        |

---

## ⚙️ How It Works

1. **User enters a company name or ticker.**
2. `data_loader.py` resolves it to a real, validated ticker via Yahoo Finance's global search — trying multiple candidates and fallbacks, and converting foreign currencies to ₹.
3. Historical price data is cleaned, and technical indicators (SMA, RSI) are calculated.
4. `model.py` trains (or loads a cached) LSTM model on the stock's closing-price history and generates a 7-day-ahead forecast, along with a suggested trading strategy (target/stop-loss).
5. `app.py` ties it all together and renders everything in `index.html`, including an interactive Plotly chart with the historical price, moving average, and forecasted trajectory.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

(Windows users can also just double-click `run.bat`.)

---

## 📁 Project Structure

```
├── app.py              # Flask routes and app entry point
├── data_loader.py       # Ticker resolution, live data fetching, currency conversion, indicators
├── model.py              # LSTM model training/loading and 7-day forecasting logic
├── templates/
│   └── index.html        # Dashboard UI (Plotly charts, forecast cards, tables)
├── requirements.txt
└── run.bat                # Windows launcher script
```

---

]
