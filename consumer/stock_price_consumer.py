from confluent_kafka import Consumer, KafkaException
import json
import psycopg2
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

KAFKA_CONFIG = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    "group.id": "stock-price-consumer-group",
    "auto.offset.reset": "earliest"
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

consumer = Consumer(KAFKA_CONFIG)
consumer.subscribe(["stock-prices"])

print("Listening for stock price updates")

def insert_or_update_stock_price(stock_data):

    try:
        stock_timestamp = datetime.datetime.fromtimestamp(stock_data["timestamp"])
        query = """
        INSERT INTO stock_prices (symbol, current_price, open_price, high_price, low_price, prev_close, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol) 
        DO UPDATE SET 
            current_price = EXCLUDED.current_price,
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            prev_close = EXCLUDED.prev_close,
            timestamp = EXCLUDED.timestamp;
        """
        cur.execute(query, (
            stock_data["symbol"], stock_data["current_price"], stock_data["open_price"],
            stock_data["high_price"], stock_data["low_price"], stock_data["prev_close"],
            stock_timestamp
        ))
        conn.commit()

    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaException._PARTITION_EOF:
                continue
            else:
                print(f"Consumer error: {msg.error()}")
                break

        stock_data = json.loads(msg.value().decode("utf-8"))

        print(f"\nStock: {stock_data['symbol']}")
        print(f"Current Price: {stock_data['current_price']}")
        print(f"Open: {stock_data['open_price']}, High: {stock_data['high_price']}, Low: {stock_data['low_price']}")
        print(f"Previous Close: {stock_data['prev_close']}")
        print(f"Timestamp: {stock_data['timestamp']}")

        insert_or_update_stock_price(stock_data)

except KeyboardInterrupt:
    print("\nConsumer stopped.")

finally:
    consumer.close()
    cur.close()
    conn.close()

