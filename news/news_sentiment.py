import tensorflow as tf
import numpy as np
import psycopg2
from transformers import TFBertForSequenceClassification, BertTokenizer
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import os

load_dotenv()

# Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Load FinBERT model and tokenizer
model_path = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)

def fetch_news():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)
    
    cur.execute("SELECT id, headline, summary FROM news")
    news_data = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return news_data


def analyze_sentiment(news_list):
    results = []
    
    for news in news_list:
        news_id = news["id"]
        text = news["headline"] + " " + news["summary"]  # Combine headline + summary
        
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='tf')
        
        outputs = model(**inputs)

        scores = tf.nn.tanh(outputs.logits).numpy()[0]

        raw_score = scores[2] - scores[0]
        sentiment_score = np.sign(raw_score) * np.log1p(abs(raw_score))

        results.append((news_id, sentiment_score))
    
    return results

def update_sentiment_scores(sentiments):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    for news_id, sentiment_score in sentiments:
        cur.execute("UPDATE news SET sentiment_score = %s WHERE id = %s", (float(sentiment_score), news_id))
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    news_list = fetch_news()
    
    if news_list:
        sentiments = analyze_sentiment(news_list)
        update_sentiment_scores(sentiments)
        print("Sentiment scores updated successfully!")
    else:
        print("No new news articles to process.")
