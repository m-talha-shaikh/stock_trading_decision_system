import finnhub

def fetch_stock_price(api_key, stock_symbol):

    finnhub_client = finnhub.Client(api_key=api_key)
    
    try:
        stock_data = finnhub_client.quote(stock_symbol)
        return {
            "symbol": stock_symbol,
            "current_price": stock_data["c"],
            "open_price": stock_data["o"],
            "high_price": stock_data["h"],
            "low_price": stock_data["l"],
            "prev_close": stock_data["pc"],
            "timestamp": stock_data["t"]
        }
    except Exception as e:
        print(f"Error fetching stock price for {stock_symbol}: {e}")
        return None
