import yfinance as yf

data = yf.download("TSLA", period="5d")

print(data)