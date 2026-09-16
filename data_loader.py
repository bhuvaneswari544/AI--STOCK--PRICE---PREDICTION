import os
import time
import requests
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

CACHE = {}
CACHE_TTL = 180  # 3 minutes cache for live data

def get_inr_rate(source_currency='USD'):
    """Fetches real-time exchange rate to convert foreign currencies to INR (₹)."""
    curr = str(source_currency).upper().strip()
    if curr in ['INR', '₹']:
        return 1.0
    
    fx_ticker = f"{curr}INR=X"
    try:
        data = yf.download(fx_ticker, period='2d', interval='1d', progress=False)
        if data is not None and not data.empty:
            close_col = data['Close']
            rate = float(close_col.iloc[-1, 0] if isinstance(close_col, pd.DataFrame) else close_col.iloc[-1])
            if rate > 0:
                return rate
    except Exception:
        pass

    fallbacks = {'USD': 84.0, 'EUR': 91.5, 'GBP': 108.0, 'JPY': 0.55, 'CAD': 61.5, 'AUD': 54.0}
    return fallbacks.get(curr, 84.0)

def search_global_companies(query, max_results=10):
    """
    Searches across all public companies in the world across global exchanges.
    Returns structured list of matching companies with name, symbol, and exchange.
    """
    q = query.strip()
    if not q:
        return []

    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(q)}&quotesCount={max_results}&newsCount=0&enableFuzzyQuery=true"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

    results = []
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            quotes = resp.json().get('quotes', [])
            for item in quotes:
                sym = item.get('symbol', '')
                name = item.get('shortname') or item.get('longname') or sym
                exchange = item.get('exchange', 'Global')
                quote_type = item.get('quoteType', 'EQUITY')
                if sym and quote_type in ['EQUITY', 'ETF', 'MUTUALFUND', 'CRYPTOCURRENCY', 'CURRENCY']:
                    results.append({
                        'symbol': sym,
                        'name': name,
                        'exchange': exchange,
                        'type': quote_type.title()
                    })
    except Exception as e:
        print(f"[Global Search Error] {e}")

    if not results and len(q) >= 1:
        results.append({
            'symbol': q.upper(),
            'name': q.upper(),
            'exchange': 'Market',
            'type': 'Stock'
        })

    return results

def resolve_company_ticker(query):
    """
    Resolves any company name in the world or ticker into the official Yahoo Finance ticker.
    """
    q = query.strip()
    if q.upper().endswith('.NS') or q.upper().endswith('.BO'):
        return q.upper(), q.upper(), 'NSE/BSE'

    matches = search_global_companies(q, max_results=8)
    if matches:
        # Prefer Indian exchange (.NS / .BO) if available
        for item in matches:
            sym = item['symbol']
            if sym.endswith('.NS') or sym.endswith('.BO'):
                return sym, item['name'], item['exchange']
        
        top = matches[0]
        return top['symbol'], top['name'], top['exchange']

    return q.upper(), q.upper(), 'Market'

def _clean_df(raw_df):
    """Flattens any MultiIndex columns and cleans NaN rows."""
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
        df.columns = flat_cols

    renames = {}
    for c in df.columns:
        cl = str(c).lower().replace(" ", "")
        if cl in ['open', 'high', 'low', 'close', 'adjclose', 'volume']:
            renames[c] = 'Adj Close' if cl == 'adjclose' else cl.capitalize()
    if renames:
        df = df.rename(columns=renames)
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    return df.dropna(subset=['Close'])

def get_stock_data(query, period='1y'):
    """
    Main data fetching pipeline: downloads prices, converts to ₹,
    and returns historical data, summary stats, and chart datasets.
    """
    ticker, company_name, exchange = resolve_company_ticker(query)
    cache_key = f"{ticker}_{period}"
    now = time.time()

    if cache_key in CACHE and (now - CACHE[cache_key][0] < CACHE_TTL):
        return CACHE[cache_key][1]

    raw_df = yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False)
    if raw_df is None or raw_df.empty or len(raw_df) < 5:
        # Direct chart query fallback
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={period}&interval=1d"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        data = r.json()['chart']['result'][0]
        timestamps = [datetime.datetime.fromtimestamp(ts) for ts in data['timestamp']]
        quotes = data['indicators']['quote'][0]
        df = pd.DataFrame({
            'Open': quotes.get('open', []),
            'High': quotes.get('high', []),
            'Low': quotes.get('low', []),
            'Close': quotes.get('close', []),
            'Volume': quotes.get('volume', [])
        }, index=pd.to_datetime(timestamps)).dropna(subset=['Close'])
    else:
        df = _clean_df(raw_df)

    if df.empty:
        raise ValueError(f"Could not find stock data for '{query}'.")

    # Currency conversion to Indian Rupees (₹)
    is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
    source_curr = 'INR' if is_indian else 'USD'
    try:
        t_info = yf.Ticker(ticker).fast_info
        if hasattr(t_info, 'currency') and t_info.currency:
            source_curr = t_info.currency
    except Exception:
        pass

    inr_multiplier = get_inr_rate(source_curr)

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = (df[col] * inr_multiplier).round(2)

    # Technical Indicators
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean().round(2)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50.0).round(1)

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    curr_price = float(latest['Close'])
    prev_close = float(prev['Close'])
    price_change = round(curr_price - prev_close, 2)
    price_change_pct = round((price_change / prev_close) * 100, 2) if prev_close != 0 else 0.0

    # Clean JSON format for Plotly.js
    chart_data = {
        'dates': [d.strftime('%Y-%m-%d') for d in df.index],
        'open': df['Open'].tolist(),
        'high': df['High'].tolist(),
        'low': df['Low'].tolist(),
        'close': df['Close'].tolist(),
        'sma20': df['SMA_20'].tolist(),
        'volume': [int(v) if not pd.isna(v) else 0 for v in df['Volume']]
    }

    # Format previous session records for the table (most recent on top)
    history_records = []
    for dt, row in df.tail(30).iloc[::-1].iterrows():
        history_records.append({
            'date': dt.strftime('%d %b %Y'),
            'iso_date': dt.strftime('%Y-%m-%d'),
            'open': f"₹{row['Open']:,.2f}",
            'high': f"₹{row['High']:,.2f}",
            'low': f"₹{row['Low']:,.2f}",
            'close': f"₹{row['Close']:,.2f}",
            'volume': f"{int(row['Volume']):,}" if not pd.isna(row['Volume']) else '0'
        })

    summary = {
        'query': query,
        'ticker': ticker,
        'company_name': company_name,
        'exchange': exchange,
        'currency': '₹',
        'is_converted': not is_indian,
        'original_curr': source_curr,
        'current_price': f"{curr_price:,.2f}",
        'raw_price': curr_price,
        'prev_close': f"{prev_close:,.2f}",
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'day_high': f"{float(latest['High']):,.2f}",
        'day_low': f"{float(latest['Low']):,.2f}",
        'high_52w': f"{float(df['High'].max()):,.2f}",
        'low_52w': f"{float(df['Low'].min()):,.2f}",
        'volume': f"{int(latest['Volume']):,}" if not pd.isna(latest['Volume']) else '0',
        'rsi': float(latest['RSI']),
        'period': period,
        'last_updated': df.index[-1].strftime('%d %b %Y')
    }

    result = {
        'df': df,
        'summary': summary,
        'chart_data': chart_data,
        'history_records': history_records
    }

    CACHE[cache_key] = (now, result)
    return result