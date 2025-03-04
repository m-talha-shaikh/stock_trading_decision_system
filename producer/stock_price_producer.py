from confluent_kafka import Producer
import json
import time
from stock_price_fetcher import fetch_stock_price
from dotenv import load_dotenv
import os

load_dotenv()

# Kafka Configuration
conf = {'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS")}
producer = Producer(conf)
TOPIC_NAME = "stock-prices"

# Stock symbols to track
stock_symbols = [
    "V", "JPM", "JNJ", "WMT", "PG", "DIS", "MA", "PFE", "HD", "BAC",
    "INTC", "XOM", "KO", "PEP", "MRK", "ABBV", "TMO", "COST", "MCD",
    "CVX", "UNH"
]

API_KEY = os.getenv("API_KEY")

def delivery_report(err, msg):

    if err:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def send_stock_prices():

    for symbol in stock_symbols:
        stock_data = fetch_stock_price(API_KEY, symbol)
        if stock_data:
            json_data = json.dumps(stock_data)
            producer.produce(TOPIC_NAME, key=symbol, value=json_data, callback=delivery_report)
    
    producer.flush()

if __name__ == "__main__":
    while True:
        send_stock_prices()
        time.sleep(240)  # Fetch every 4 min
