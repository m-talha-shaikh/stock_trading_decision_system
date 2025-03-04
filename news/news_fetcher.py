import finnhub
import psycopg2
import datetime
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()

# API Key
API_KEY = os.getenv("API_KEY")

# PostgreSQL Connection Config
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Stock Dictionary
stock_dict = {
    "V": "Visa",
    "JPM": "JPMorgan",
    "JNJ": "Johnson",
    "WMT": "Walmart",
    "PG": "Procter",
    "DIS": "Walt Disney",
    "MA": "Mastercard",
    "PFE": "Pfizer",
    "HD": "Home Depot",
    "BAC": "Bank of America",
    "INTC": "Intel",
    "XOM": "Exxon",
    "KO": "Coca Cola",
    "PEP": "Pepsi",
    "MRK": "Merck",
    "ABBV": "AbbVie",
    "TMO": "Thermo Fisher",
    "COST": "Costco",
    "MCD": "McDonald's",
    "CVX": "Chevron",
    "UNH": "UnitedHealth"
}

# Finnhub Client
finnhub_client = finnhub.Client(api_key=API_KEY)

# Read last fetch date
def get_last_fetch_date():
    try:
        with open("last_fetch.txt", "r") as file:
            print("reading")
            return file.read().strip()
    except FileNotFoundError:
        print("not reading")
        return "2025-01-01"  # Default

# Update last fetch date
def update_last_fetch_date():
    with open("last_fetch.txt", "w") as file:
        file.write(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

# Fetch news from Finnhub
def fetch_latest_news():
    last_fetch_date = get_last_fetch_date()
    latest_news = []

    for symbol, company_name in stock_dict.items():
        try:
            news = finnhub_client.company_news(symbol, _from=last_fetch_date, to=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            count = 0  

            for article in news:
                news_id = article.get("id")  
                headline = article.get("headline", "No headline")
                summary = article.get("summary", "No summary")
                source = article.get("source", "Unknown Source")  
                timestamp = article.get("datetime", 0)

                news_timestamp = datetime.fromtimestamp(timestamp, timezone.utc)

                summary = summary.encode("utf-8", "ignore").decode("utf-8")

                # Apply filtering condition
                if (symbol in headline or company_name in headline) and "transcript" not in headline.lower():
                    latest_news.append({
                        "id": news_id,  
                        "symbol": symbol,
                        "headline": headline,
                        "summary": summary,
                        "source": source,  
                        "timestamp": news_timestamp
                    })
                    count += 1

                if count >= 5:  # Limit to 5 articles per stock
                    break

        except Exception as e:
            print(f"Error fetching news for {company_name} ({symbol}): {e}")

        time.sleep(3)

    return latest_news

# Store news in PostgreSQL
def store_news_in_db(news_list):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        for news in news_list:
            query = """
            INSERT INTO news (id, symbol, headline, summary, source, sentiment_score, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
            """
            cur.execute(query, (news["id"], news["symbol"], news["headline"], news["summary"], news["source"], 0.0, news["timestamp"]))

        conn.commit()
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    news_data = fetch_latest_news()
    if news_data:
        store_news_in_db(news_data)
        update_last_fetch_date()
    else:
        print("No new news articles found.")
