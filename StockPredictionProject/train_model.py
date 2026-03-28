"""
Model training script for pre-trained models
You can run this separately to train and save models
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_and_save_model(ticker='AAPL', period='2y'):
    """Train and save models for a specific stock"""
    
    print(f"Training models for {ticker}...")
    
    # Fetch data
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    if data.empty:
        print(f"No data found for {ticker}")
        return
    
    # Prepare features
    df = data.copy()
    df['Day'] = np.arange(len(df))
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()
    
    feature_cols = ['Day', 'MA_7', 'MA_21', 'Volatility', 'Returns']
    X = df[feature_cols].values
    y = df['Close'].values
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model, f'models/lr_model_{ticker}.pkl')
    joblib.dump(rf_model, f'models/rf_model_{ticker}.pkl')
    joblib.dump(scaler, f'models/scaler_{ticker}.pkl')
    
    print(f"✅ Models saved for {ticker}")
    return True

if __name__ == "__main__":
    # Train for common stocks
    stocks = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
    
    for stock in stocks:
        try:
            train_and_save_model(stock)
        except Exception as e:
            print(f"Error training {stock}: {e}")
    
    print("Training complete!")