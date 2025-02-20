# 📈 Stock Price Prediction Using LSTM & Sentiment Analysis

## 🚀 Overview
This project predicts **Alphabet Inc. (GOOG) stock prices** by integrating **historical stock data, market trends, and sentiment analysis** from news articles and Reddit discussions. A **Long Short-Term Memory (LSTM) model** is trained on these features to forecast future stock prices.

## 📊 Data Sources
- **Stock Market Data**: Downloaded from **Yahoo Finance** (`yfinance`).
  - GOOG stock prices (Open, Close, High, Low, Volume)
  - **Technical indicators**: Moving Averages (50-day, 200-day) & RSI
  - **S&P 500 index** (for broader market trends)
- **Sentiment Data**:
  - **News Headlines**: Fetched using **NewsAPI**
  - **Reddit Posts**: Scraped from `/r/stocks` using **PRAW** (Python Reddit API Wrapper)
  - **Sentiment Analysis**: Scored using **BERT NLP Model** (`nlptown/bert-base-multilingual-uncased-sentiment`)

## 🔄 Data Processing & Feature Engineering
- **Merged stock & sentiment data** based on dates.
- **Computed technical indicators** (Moving Averages, RSI, S&P 500 trends).
- **Applied sentiment analysis** on news & Reddit headlines (scores range from 1 = negative to 5 = positive).
- **Filled missing sentiment values** with neutral sentiment (3).
- **Normalized** stock prices using **MinMaxScaler**.
- **Created time-series sequences** (30-day lookback) for LSTM model training.

## 🧠 Model Architecture: LSTM (Long Short-Term Memory)
- **Input:** Sequences of 30 days of stock & sentiment data.
- **LSTM Layers:**
  - 2 LSTM layers (50 units each, first layer returns sequences)
  - Dropout layers (to prevent overfitting)
- **Dense Layers:**
  - Fully connected layers (25 & 1 neuron for final prediction)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Training:** 80% of data used for training, 20% for validation.

## 📉 Results & Insights
- The model aims to capture stock price trends by integrating **market indicators & sentiment analysis**.
- Future improvements can include:
  - **Hyperparameter tuning** for better accuracy.
  - **Testing feature importance** (e.g., evaluating the impact of sentiment scores).
  - **Deploying the model** as a Flask API for real-time predictions.

## 📂 Installation & Usage
### 🔧 Install Dependencies
```bash
pip install pandas numpy yfinance newsapi-python praw transformers torch scikit-learn tensorflow
```

### ▶️ Run the Script
```bash
python stock_price_prediction.py
```

## 🏆 Skills Demonstrated
✅ **Machine Learning (LSTM for Time-Series Forecasting)**  
✅ **Natural Language Processing (BERT Sentiment Analysis)**  
✅ **Feature Engineering & Data Merging**  
✅ **API Data Retrieval (Yahoo Finance, NewsAPI, Reddit API)**  
✅ **Deep Learning (TensorFlow/Keras)**  

---
📌 **Author:** [Your Name]  
💡 **Feel free to contribute!** 🚀

