### Necessary Packages

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta

# for Stock and S&P 500 data download
import yfinance as yf 
from time import sleep
from sklearn.preprocessing import MinMaxScaler

# for News API article download
from newsapi import NewsApiClient

# Reddit data download
import praw

# for Sentiment analysis
import requests, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# for LSTM model training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

today = datetime.today()
end_date = datetime.today() - timedelta(days=30)
start_date = datetime.strptime("2023-01-01", "%Y-%m-%d")


### Alphabet Inc. Stock Data Download

# Alphabet Inc. ticker
ticker = "GOOG"
sleep(20)
stock_data = yf.download(ticker, start=start_date, end=today)
sleep(20)

# Calculate Moving Average and add to dataframe
stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()


# Calculate Relative Stock Index (RSI) and add to dataframe
delta = stock_data['Close'].diff()

# Seperate gains and losses
gain = np.where(delta > 0, delta, 0).flatten()
loss = np.where(delta < 0, -delta, 0).flatten()

# Flatten the gains and losses arrays to prepare for calculation
gain_series = pd.Series(gain, index=stock_data.index)
loss_series = pd.Series(loss, index=stock_data.index)

# Calculate average gains and losses
period = 14
avg_gain = gain_series.ewm(span=period, adjust=False).mean()
avg_loss = loss_series.ewm(span=period, adjust=False).mean()

# Calculate Relative Strength (RS)
rs = avg_gain / (avg_loss + 1e-10) # Ensure that nothing is divided by zero
rsi = 100 - (100 / (1 + rs))
rsi.replace([np.inf, -np.inf], np.nan, inplace=True)

stock_data["RSI"] = rsi

# Convert stock data to dataframe
df_stock = pd.DataFrame(stock_data)

df_stock.columns

# Change format of date index to mm-dd-yyyy
#df_stock.index = df_stock.index.strftime('%m-%d-%Y')
# Change back to datetime format for sorting

# Add new index column
df_stock = df_stock.reset_index()

df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock = df_stock.sort_values(by='Date', ascending=False)
# Change format of date index to mm-dd-yyyy
df_stock['Date'] = df_stock['Date'].dt.strftime('%m-%d-%Y')


df_stock.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in df_stock.columns]
df_stock = df_stock.rename(columns={"Close_GOOG":"Close", "High_GOOG":"High",
                          "Low_GOOG":"Low", "Open_GOOG":"Open",
                          "Volume_GOOG":"Volume"})

df_stock

### S&P 500 Data Download

sp500 = "^GSPC"
sleep(60)
sp500_data = yf.download(sp500, start=start_date, end=today, interval='1d')
sleep(15)

# Only keep closing price column
sp500_data = sp500_data[['Close']]
sp500_data.rename(columns={"Close": "SP_500_Close"}, inplace=True)

# Normalize closing prices
scaler_sp500 = MinMaxScaler(feature_range=(0,1))
sp500_data["SP_500_Close"] = scaler_sp500.fit_transform(sp500_data["SP_500_Close"])

df_sp500 = pd.DataFrame()
df_sp500 = pd.DataFrame(sp500_data)

# Change Date column from being index column
df_sp500 = df_sp500.reset_index()

#df_sp500.index = df_sp500.index.strftime('%m-%d-%Y')
# Change back to datetime format for sorting
df_sp500['Date'] = pd.to_datetime(df_sp500['Date'])
df_sp500 = df_sp500.sort_values(by='Date', ascending=False)

# Change date format to mm-dd-yyyy
df_sp500['Date'] = df_sp500['Date'].dt.strftime('%m-%d-%Y')

df_sp500.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in df_sp500.columns]
df_sp500 = df_sp500.rename(columns={"SP_500_Close_^GSPC":"SP_500_Close"})



df_sp500



### Merge both stock dataframes
stocks_df = df_stock.merge(df_sp500, on='Date', how='left')
stocks_df = pd.DataFrame()

stocks_df.columns

### Download News Articles for Sentiment

newsapi = NewsApiClient(api_key="e12653ac19104f128c85b4b14e00f32f")

# Find articles with Alphabet Inc. or Google in the headlines
data = newsapi.get_everything(
  q="alphabet inc." or "google", 
  language='en'
  )

articles = data['articles']
df_articles = pd.DataFrame(articles)


# Change date column format
df_articles['Date'] = pd.to_datetime(df_articles['publishedAt'])
df_articles['Date'] = df_articles['Date'].dt.strftime('%m-%d-%Y')

df_articles = df_articles.rename(columns={"title":"Title"})

headlines = df_articles[['Date','Title']]

# Change back to datetime format for sorting
headlines['Date'] = pd.to_datetime(headlines['Date'])

headlines

### Get Reddit data


reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent
                     )

subreddit = reddit.subreddit("stocks")

def fetch_reddit_posts():
    subreddit = reddit.subreddit("stocks")
    posts = [post for post in subreddit.search("Alphabet Inc. OR GOOGL", limit=500)]

    # Collect data
    reddit_data = [{"date": post.created_utc, "title": post.title} for post in posts]
    return reddit_data

reddit_data = fetch_reddit_posts()

#reddit_df = pd.DataFrame()
reddit_df = pd.DataFrame(reddit_data)

reddit_df['date'] = pd.to_datetime(reddit_df['date'], unit='s')   

reddit_df = reddit_df[
   (reddit_df['date'] >= start_date)
     & (reddit_df['date'] <= end_date)]

reddit_df['date'] = reddit_df['date'].dt.strftime('%m-%d-%Y')
reddit_df = reddit_df.rename(columns={"date":"Date", "title":"Title"})

# Change back to datetime format for sorting
reddit_df['Date'] = pd.to_datetime(reddit_df['Date'])

reddit_df

### Merge reddit and newsapi data

headlines = headlines.sort_values(by='Date', ascending=False)
reddit_df = reddit_df.sort_values(by='Date', ascending=False)

reddit_df
headlines
headlines.tail()

df_sentiment = pd.DataFrame()

# Outer join so no data is lost
df_sentiment = headlines.merge(reddit_df, on='Date', how='outer')

# Fill blank columns in the Title_x, or NewsAPI, column
df_sentiment['Title'] = df_sentiment['Title_x'].fillna(df_sentiment['Title_y'])

# Drop blank columns
df_sentiment.drop(columns=['Title_x', 'Title_y'], inplace=True)

# Change date format to mm-dd-yyyy
df_sentiment['Date'] = df_sentiment['Date'].dt.strftime('%m-%d-%Y')

df_sentiment

### Calculate sentiment scores

# Instantiate model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# Find sentiment score
def sentiment_score(article):
  tokens = tokenizer.encode(article, return_tensors='pt')
  result = model(tokens)
  return int(torch.argmax(result.logits))+1

df_sentiment['Sentiment_Score'] = df_sentiment['Title'].apply(lambda x: sentiment_score(x[:512])) # NLP model only allows to 512 value


### Merge all dataframes
final_df = pd.DataFrame()

final_df = stocks_df.merge(df_sentiment, on='Date', how='left')

final_df.columns
final_df = final_df.drop(columns="Title")

final_df['Sentiment_Score'] = final_df['Sentiment_Score'].fillna(3)


### Prepare data for LSTM
final_df = final_df.drop(columns=['Date'])

time_steps = 30

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i + time_steps].values)  # Features
        y.append(data.iloc[i + time_steps]["Close"])
    return np.array(X), np.array(y)  # Target variable

# To create sequences
X, y = create_sequences(final_df, time_steps)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train.shape
X_test.shape
### Train model for LSTM analaysis

lstm_model = Sequential([
   LSTM(units=50, return_sequences=True, input_shape=(time_steps, X.shape[2])),
   Dropout(0.2),
   LSTM(units=50, return_sequences=False),
   Dropout(0.2),
   Dense(units=25),
   Dense(units=1)
])

# Compile model
lstm_model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32,
                         validation_data=(X_test, y_test))



