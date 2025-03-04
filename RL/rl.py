import time
import psycopg2
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from decimal import Decimal
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


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    """Convert text into a BERT embedding."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def fetch_stock_data():
    """Fetch latest stock prices."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
    SELECT s.symbol, sp.current_price
    FROM stocks s
    JOIN stock_prices sp ON s.symbol = sp.symbol;
    """
    
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()

    return {row[0]: {"price": row[1]} for row in data}

def fetch_news_data():
    """Fetch news headlines & summaries for each stock and compute aggregated embeddings."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
    SELECT symbol, headline, summary, sentiment_score FROM news;
    """
    
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()

    news_embeddings = {}

    for row in data:
        symbol, headline, summary, sentiment_score = row
        text = headline + " " + summary
        embedding = get_bert_embedding(text)

        if symbol not in news_embeddings:
            news_embeddings[symbol] = {"embeddings": [], "sentiment_scores": []}

        news_embeddings[symbol]["embeddings"].append(embedding)
        news_embeddings[symbol]["sentiment_scores"].append(sentiment_score)

    for symbol in news_embeddings:
        embeddings = np.array(news_embeddings[symbol]["embeddings"])
        sentiments = np.array(news_embeddings[symbol]["sentiment_scores"])

        if len(sentiments) > 0:
            weights = sentiments / np.sum(sentiments) if np.sum(sentiments) != 0 else np.ones_like(sentiments) / len(sentiments)
            weighted_embedding = np.average(embeddings, axis=0, weights=weights)

        else:
            weighted_embedding = np.mean(embeddings, axis=0)
        
        news_embeddings[symbol] = weighted_embedding.squeeze()

    return news_embeddings

class RLTrader:

    def __init__(self, symbols):
        self.q_table = {symbol: np.random.randn(768 + 1, 3) for symbol in symbols}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate

    def choose_action(self, symbol, news_embedding, price_change):
        price_change = float(price_change) if isinstance(price_change, Decimal) else price_change
        
        state = np.concatenate([news_embedding, [price_change]])
        
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])

        q_values = np.dot(state, self.q_table[symbol])  
        return np.argmax(q_values)

    def update_q_values(self, symbol, action, reward):
        """Update Q-values based on reward."""
        self.q_table[symbol][action] += self.alpha * (float(reward) - self.q_table[symbol][action])


stock_data = fetch_stock_data()
news_embeddings = fetch_news_data()
trader = RLTrader(stock_data.keys())

print("Initial stock & news data fetched.")
time.sleep(10)

new_stock_data = fetch_stock_data()
print("Updated stock data fetched.")

for symbol in stock_data:
    stock_data[symbol]["old_price"] = stock_data[symbol]["price"]
    stock_data[symbol]["new_price"] = new_stock_data[symbol]["price"]

# Trading Loop
balance = 10000
positions = {symbol: 0 for symbol in stock_data}
true_labels, predicted_labels = [], []

for i in range(10):  # 5 trading iterations
    print("\n--- Trading Cycle ---")

    for symbol in stock_data:
        old_price = stock_data[symbol]["old_price"]
        new_price = stock_data[symbol]["new_price"]
        price_change = new_price - old_price
        news_embedding = news_embeddings.get(symbol, np.zeros(768))  # Default zero vector if no news

        action = trader.choose_action(symbol, news_embedding, price_change)
        action_label = 1 if action == 0 else (-1 if action == 2 else 0)
        true_label = 1 if price_change > 0 else (-1 if price_change < 0 else 0)

        predicted_labels.append(action_label)
        true_labels.append(true_label)

        # Reward Calculation
        reward = (price_change / old_price) * 100  
        reward = float(reward)
        reward -= 0.1
        
        
        trader.update_q_values(symbol, action, reward)

        if action == 0 and balance >= new_price:  # BUY
            positions[symbol] += 1
            balance -= new_price
        elif action == 2 and positions[symbol] > 0:  # SELL
            positions[symbol] -= 1
            balance += new_price

        print(f"Stock: {symbol}, Action: {['BUY', 'HOLD', 'SELL'][action]}, Price Change: {price_change}, Reward: {reward}")

    print("Sleeping...")
    print(i)
    time.sleep(10)
    updated_stock_data = fetch_stock_data()

    for symbol in stock_data:
        stock_data[symbol]["old_price"] = stock_data[symbol]["new_price"]
        stock_data[symbol]["new_price"] = updated_stock_data[symbol]["price"]

# Evaluation
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)

print("\n--- Performance Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
