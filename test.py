import yfinance as yf

ticker = yf.Ticker("AAPL")

# available expirations
print(ticker.options)

# current option chain for one expiry
chain = ticker.option_chain(ticker.options[0])

calls = chain.calls
puts = chain.puts

calls[["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]].head()