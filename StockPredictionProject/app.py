"""
StockVision AI - Advanced Stock Prediction Dashboard
Professional stock analysis with multiple ML models and user authentication
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="StockVision AI | Stock Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for Better Text Visibility
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #ffffff !important;
        opacity: 0.95;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .stSidebar {
        background-color: #ffffff;
    }
    
    /* Sidebar text */
    .stSidebar .stMarkdown, 
    .stSidebar label, 
    .stSidebar p, 
    .stSidebar div,
    .stSidebar span,
    .stSidebar h1,
    .stSidebar h2,
    .stSidebar h3,
    .stSidebar h4 {
        color: #1e3c72 !important;
        font-weight: 600 !important;
    }
    
    /* Make selected values in selectbox white on blue background */
    .stSidebar .stSelectbox div[data-baseweb="select"] {
        background-color: #1e3c72 !important;
        border-radius: 8px;
    }
    
    .stSidebar .stSelectbox div[data-baseweb="select"] div {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Make dropdown options white when selected */
    .stSidebar .stSelectbox ul li[aria-selected="true"] {
        background-color: #2a5298 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar info box */
    .stSidebar .stAlert {
        background-color: #e8f0fe;
        color: #1e3c72 !important;
    }
    
    /* Sidebar checkbox labels */
    .stSidebar .stCheckbox label {
        color: #1e3c72 !important;
        font-weight: 500 !important;
    }
    
    /* Button styling - white text */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff !important;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        transition: transform 0.3s, box-shadow 0.3s;
        border-left: 4px solid #2a5298;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        color: #1e3c72 !important;
        font-size: 0.9rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #000000 !important;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 0.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #2c3e50 !important;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: all 0.3s;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #1e3c72 !important;
        background: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] button {
        color: #ffffff !important;
    }
    
    /* Prediction cards - ALL TEXT WHITE */
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-card h2, 
    .prediction-card h3, 
    .prediction-card p,
    .prediction-card span {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    .model-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .model-card:hover {
        border-color: #2a5298;
        box-shadow: 0 4px 12px rgba(26, 60, 114, 0.1);
    }
    
    .model-card h4 {
        color: #1e3c72 !important;
        font-weight: 800;
        font-size: 1.1rem;
    }
    
    .model-card h2 {
        color: #2a5298 !important;
        font-weight: 800;
        font-size: 1.8rem;
    }
    
    .model-card p {
        color: #000000 !important;
        font-weight: 500;
    }
    
    .model-card small {
        color: #6c757d !important;
        font-weight: 500;
    }
    
    /* Signal badges */
    .buy-signal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724 !important;
        padding: 0.75rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 800;
        font-size: 1.1rem;
        border: 2px solid #28a745;
    }
    
    .buy-signal h3, .buy-signal p {
        color: #155724 !important;
        font-weight: 700;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24 !important;
        padding: 0.75rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 800;
        font-size: 1.1rem;
        border: 2px solid #dc3545;
    }
    
    .sell-signal h3, .sell-signal p {
        color: #721c24 !important;
        font-weight: 700;
    }
    
    .neutral-signal {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        color: #856404 !important;
        padding: 0.75rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 800;
        font-size: 1.1rem;
        border: 2px solid #ffc107;
    }
    
    .neutral-signal h3, .neutral-signal p {
        color: #856404 !important;
        font-weight: 700;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #2a5298;
    }
    
    .stAlert p {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* DataFrame styling */
    .dataframe {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .dataframe td, .dataframe th {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem 1rem;
        color: #000000 !important;
        font-weight: 500;
    }
    
    .stTextInput label {
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }
    
    /* Select boxes */
    .stSelectbox label {
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #1e3c72, #2a5298, #4a6fa5);
        margin: 1.5rem 0;
        border-radius: 3px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: white;
        border-radius: 15px;
    }
    
    .footer p {
        color: #1e3c72 !important;
        font-weight: 600;
    }
    
    /* General text */
    p, li, div, span, label, .stMarkdown {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    h1, h2, h3, h4 {
        color: #1e3c72 !important;
        font-weight: 800;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #1e3c72 !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }
    
    /* Success/Info/Warning text */
    .stSuccess p, .stInfo p, .stWarning p {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Login page specific */
    .login-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Make stock names in dropdown white */
    .stSelectbox ul li {
        color: #000000 !important;
    }
    
    /* Make all text in login page white on blue background */
    .main-header .stMarkdown, 
    .main-header p, 
    .main-header h1 {
        color: #ffffff !important;
    }
    
    /* Stock names in sidebar dropdown - make them white when selected */
    .stSidebar .stSelectbox div[data-baseweb="select"] div {
        color: #ffffff !important;
    }
    
    /* Make all prediction card text white */
    div[data-testid="stMarkdownContainer"] .prediction-card {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Authentication functions
def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                users_data = json.load(f)
                if isinstance(users_data, dict):
                    return users_data
                else:
                    return {}
        return {}
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")
        return {}

def authenticate(username, password):
    """Authenticate user with simple password check"""
    users = load_users()
    if username in users and users[username] == password:
        return True
    return False

def register_user(username, password):
    """Register new user"""
    users = load_users()
    if username in users:
        return False
    users[username] = password
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
        return True
    except:
        return False

# Stock Predictor Class
class AdvancedStockPredictor:
    """Advanced stock prediction with multiple ML models"""
    
    def __init__(self, ticker, data=None):
        self.ticker = ticker
        self.data = data
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_features(self, df):
        """Advanced feature engineering"""
        df = df.copy()
        
        # Price-based features
        df['Day'] = np.arange(len(df))
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility features
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Momentum features
        df['Returns_1d'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)
        
        # Price position features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Volume features
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features
        feature_cols = ['Day', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 
                       'Volatility_5', 'Volatility_20', 'Returns_1d', 'Returns_5d',
                       'Returns_20d', 'High_Low_Ratio', 'Close_Open_Ratio',
                       'Volume_Ratio', 'RSI', 'BB_Upper', 'BB_Lower']
        
        X = df[feature_cols].values
        y = df['Close'].values
        
        return X, y, df, feature_cols
    
    def train_selected_models(self, selected_models):
        """Train only selected ML models"""
        try:
            X, y, df_features, feature_cols = self.prepare_features(self.data)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            
            # Dictionary of models
            models_dict = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
                'SVR': SVR(kernel='rbf', C=100, gamma='auto')
            }
            
            # Train only selected models
            for model_name in selected_models:
                if model_name in models_dict:
                    model = models_dict[model_name]
                    model.fit(X_train_scaled, y_train)
                    results[model_name] = self.evaluate_model(model, X_test_scaled, y_test, X_train_scaled, y_train, X, y)
                    
                    # Add feature importance for tree-based models
                    if model_name in ['Random Forest', 'Gradient Boosting']:
                        results[model_name]['feature_importance'] = model.feature_importances_
            
            # Add features list
            results['features'] = feature_cols
            
            return results, df_features.index[split_idx:], y_test
            
        except Exception as e:
            st.error(f"Model training error: {str(e)}")
            return None, None, None
    
    def evaluate_model(self, model, X_test, y_test, X_train, y_train, X_full, y_full):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        # Next day prediction
        last_features = X_full[-1].reshape(1, -1)
        last_features_scaled = self.scaler.transform(last_features)
        next_day_pred = model.predict(last_features_scaled)[0]
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        return {
            'model': model,
            'predictions': predictions,
            'next_day': next_day_pred,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

def get_stock_data(ticker, period='2y'):
    """Fetch stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None, None
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

def create_advanced_charts(data, ticker):
    """Create advanced visualization charts with larger size"""
    
    # Create subplots with larger size
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Price with Moving Averages', 'Trading Volume', 
                       'RSI Indicator', 'Bollinger Bands',
                       'Price Distribution', 'Returns Distribution'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Price with Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                            name='Close', line=dict(color='#1e3c72', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(20).mean(),
                            name='MA 20', line=dict(color='#dc3545', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(),
                            name='MA 50', line=dict(color='#28a745', width=1, dash='dash')), row=1, col=1)
    
    # 2. Volume
    colors = ['#dc3545' if data['Close'].iloc[i] < data['Close'].iloc[i-1] else '#28a745' 
              for i in range(1, len(data))]
    colors.insert(0, '#28a745')
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                        name='Volume', marker_color=colors), row=1, col=2)
    
    # 3. RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', 
                            line=dict(color='#ff6b6b', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#dc3545", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#28a745", row=2, col=1)
    
    # 4. Bollinger Bands
    bb_middle = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', 
                            line=dict(color='#1e3c72')), row=2, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=bb_upper, name='BB Upper', 
                            line=dict(color='#dc3545', dash='dash')), row=2, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=bb_lower, name='BB Lower', 
                            line=dict(color='#28a745', dash='dash')), row=2, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=bb_middle, name='BB Middle', 
                            line=dict(color='#ffc107', dash='dash')), row=2, col=2)
    
    # 5. Price Distribution
    fig.add_trace(go.Histogram(x=data['Close'], nbinsx=30, 
                              marker_color='#1e3c72', name='Price Dist'), row=3, col=1)
    
    # 6. Returns Distribution
    returns = data['Close'].pct_change().dropna() * 100
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, 
                              marker_color='#2a5298', name='Returns Dist'), row=3, col=2)
    
    fig.update_layout(height=1000, title_text=f"{ticker} - Advanced Technical Analysis", 
                     showlegend=True, template='plotly_white',
                     title_font_size=20)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=2, col=2)
    
    return fig

def create_prediction_comparison(results, dates, y_test):
    """Create prediction comparison chart with larger size"""
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(x=dates, y=y_test, name='Actual',
                            line=dict(color='#000000', width=3)))
    
    # Add predictions from each model
    colors = ['#1e3c72', '#2a5298', '#4a6fa5', '#6c8cbf', '#8fa9d9', '#b2c6f2']
    for i, (model_name, result) in enumerate(results.items()):
        if model_name != 'features':
            fig.add_trace(go.Scatter(x=dates, y=result['predictions'], 
                                    name=model_name,
                                    line=dict(color=colors[i % len(colors)], 
                                             width=2, dash='dash')))
    
    fig.update_layout(title='Model Predictions Comparison',
                     xaxis_title='Date',
                     yaxis_title='Price (USD)',
                     template='plotly_white',
                     height=600,
                     hovermode='x unified',
                     title_font_size=18)
    
    return fig

def create_error_heatmap(results):
    """Create model error comparison heatmap"""
    models = []
    error_data = []
    
    for model_name, result in results.items():
        if model_name != 'features' and 'mae' in result:
            models.append(model_name)
            error_data.append([result['mae'], result['rmse'], result['mape']])
    
    if error_data:
        fig = go.Figure(data=go.Heatmap(
            z=error_data,
            x=['MAE', 'RMSE', 'MAPE (%)'],
            y=models,
            colorscale='Viridis',
            text=np.round(error_data, 2),
            texttemplate='%{text}',
            textfont={"size": 12, "color": "white"}
        ))
        
        fig.update_layout(title='Model Error Comparison',
                         xaxis_title='Error Metrics',
                         yaxis_title='Models',
                         height=500,
                         title_font_size=18)
        
        return fig
    return None

def generate_trading_recommendation(results, current_price):
    """Generate comprehensive trading recommendation"""
    recommendations = []
    weights = []
    
    for model_name, result in results.items():
        if model_name != 'features' and 'next_day' in result:
            pred_change = ((result['next_day'] - current_price) / current_price) * 100
            # Weight by R² score
            weight = max(0, result['r2'])
            recommendations.append(pred_change)
            weights.append(weight)
    
    if recommendations:
        # Weighted average prediction
        weighted_pred_change = np.average(recommendations, weights=weights)
        
        if weighted_pred_change > 2:
            signal = "STRONG BUY"
            signal_class = "buy-signal"
            confidence = "High"
        elif weighted_pred_change > 0.5:
            signal = "BUY"
            signal_class = "buy-signal"
            confidence = "Medium"
        elif weighted_pred_change > -0.5:
            signal = "HOLD"
            signal_class = "neutral-signal"
            confidence = "Low"
        elif weighted_pred_change > -2:
            signal = "SELL"
            signal_class = "sell-signal"
            confidence = "Medium"
        else:
            signal = "STRONG SELL"
            signal_class = "sell-signal"
            confidence = "High"
        
        return {
            'signal': signal,
            'class': signal_class,
            'expected_change': weighted_pred_change,
            'confidence': confidence,
            'next_day_price': current_price * (1 + weighted_pred_change/100)
        }
    
    return None

def login_page():
    """Display login page"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 StockVision AI</h1>
        <p>Intelligent Stock Predictions Powered by Advanced Machine Learning</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Real-time Analysis • 6 ML Models • Professional Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### 🔐 Welcome Back")
        st.markdown("Login to access your personalized dashboard")
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            if st.button("Login", key="login_btn", use_container_width=True):
                if authenticate(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        with tab2:
            new_username = st.text_input("Username", key="reg_username", placeholder="Choose a username")
            new_password = st.text_input("Password", type="password", key="reg_password", placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            if st.button("Register", key="reg_btn", use_container_width=True):
                if not new_username or not new_password:
                    st.error("❌ Please fill all fields")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif register_user(new_username, new_password):
                    st.success("✅ Registration successful! Please login.")
                else:
                    st.error("❌ Username already exists")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📌 Demo Credentials")
        st.markdown("**Username:** sahanasree | **Password:** 123")
        st.markdown("**Username:** Sahanasree | **Password:** Sahana")

def main_app():
    """Main application after authentication"""
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>📊 StockVision AI</h1>
                <p>Intelligent Stock Predictions Powered by Advanced Machine Learning</p>
            </div>
            <div>
                <p style="color: white; margin: 0; font-size: 1.1rem; font-weight: 600;">👋 Welcome, {st.session_state['username']}!</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Stock Selection")
        st.markdown("---")
        
        # Popular stock tickers with dropdown
        popular_stocks = {
            'Select a stock...': '',
            'Apple Inc. (AAPL)': 'AAPL',
            'Microsoft Corp. (MSFT)': 'MSFT',
            'Google (GOOGL)': 'GOOGL',
            'Amazon.com Inc. (AMZN)': 'AMZN',
            'Tesla Inc. (TSLA)': 'TSLA',
            'Meta Platforms (META)': 'META',
            'NVIDIA Corp. (NVDA)': 'NVDA',
            'Netflix Inc. (NFLX)': 'NFLX',
            'The Walt Disney Co. (DIS)': 'DIS',
            'Coca-Cola Co. (KO)': 'KO',
            'Custom Ticker...': 'custom'
        }
        
        stock_option = st.selectbox(
            "Select Stock",
            list(popular_stocks.keys())
        )
        
        if stock_option == 'Custom Ticker...':
            ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
        else:
            ticker = popular_stocks[stock_option]
            if ticker and ticker != 'custom':
                st.info(f"✅ Selected: {stock_option}")
        
        if not ticker:
            st.warning("⚠️ Please select a stock")
            return
        
        period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        st.markdown("---")
        st.markdown("## 🤖 Select ML Models")
        st.markdown("Choose the models you want to use for predictions:")
        
        # Model selection checkboxes
        model_options = {
            'Linear Regression': True,
            'Ridge Regression': True,
            'Lasso Regression': False,
            'Random Forest': True,
            'Gradient Boosting': True,
            'SVR': False
        }
        
        selected_models = []
        for model_name, default_value in model_options.items():
            if st.checkbox(model_name, value=default_value):
                selected_models.append(model_name)
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model")
            return
        
        st.markdown(f"**✅ Selected Models:** {len(selected_models)}")
        
        st.markdown("---")
        st.markdown("## 📊 What You Can See Here:")
        st.markdown("""
        - ✅ Technical Indicators
        - ✅ Moving Averages (5,10,20,50)
        - ✅ RSI & Bollinger Bands
        - ✅ Volume Analysis
        - ✅ Price Patterns
        - ✅ Risk Metrics
        """)
        
        analyze_btn = st.button("🚀 Analyze Stock", use_container_width=True)
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['authenticated'] = False
            st.rerun()
    
    # Main content
    if analyze_btn:
        with st.spinner(f"Fetching and analyzing {ticker}..."):
            data, info = get_stock_data(ticker, period)
            
            if data is not None and not data.empty:
                st.session_state['stock_data'] = data
                st.session_state['stock_info'] = info
                st.session_state['ticker'] = ticker
                st.success(f"✅ Successfully fetched data for {ticker}")
                
                # Train selected models
                with st.spinner(f"Training {len(selected_models)} machine learning models..."):
                    predictor = AdvancedStockPredictor(ticker, data)
                    results, test_dates, y_test = predictor.train_selected_models(selected_models)
                    
                    if results:
                        st.session_state['ml_results'] = results
                        st.session_state['test_dates'] = test_dates
                        st.session_state['y_test'] = y_test
                        st.success(f"✅ {len(selected_models)} models trained successfully!")
                    else:
                        st.error("❌ Model training failed")
            else:
                st.error(f"❌ No data found for ticker {ticker}")
                return
    
    # Display results if available
    if 'stock_data' in st.session_state:
        data = st.session_state['stock_data']
        info = st.session_state['stock_info']
        ticker = st.session_state['ticker']
        
        # Current metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">💰 Current Price</div>
                <div class="metric-value">${current_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#28a745" if price_change >= 0 else "#dc3545"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📊 Daily Change</div>
                <div class="metric-value" style="color: {color};">{price_change:+.2f} ({price_change_pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📈 Volume</div>
                <div class="metric-value">{data['Volume'].iloc[-1]:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🏢 Market Cap</div>
                <div class="metric-value">${info.get('marketCap', 0)/1e9:.1f}B</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs with updated names
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", "📈 Technical Analysis", "🤖 Model Insights", 
            "🔮 Price Forecast", "📉 Risk Analytics", "🏆 Performance Summary"
        ])
        
        # Tab 1: Dashboard
        with tab1:
            st.markdown("## 📋 Stock Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏢 Company Details")
                st.write(f"**Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                
            with col2:
                st.markdown("### 📊 Trading Information")
                st.write(f"**52W High:** ${info.get('fiftyTwoWeekHigh', 0):.2f}")
                st.write(f"**52W Low:** ${info.get('fiftyTwoWeekLow', 0):.2f}")
                st.write(f"**50-Day Avg:** ${info.get('fiftyDayAverage', 0):.2f}")
                st.write(f"**200-Day Avg:** ${info.get('twoHundredDayAverage', 0):.2f}")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            # Recent data
            st.markdown("### 📊 Recent Price Data")
            recent_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
            recent_data.index = recent_data.index.strftime('%Y-%m-%d')
            st.dataframe(recent_data, use_container_width=True)
        
        # Tab 2: Technical Analysis
        with tab2:
            st.markdown("## 📈 Advanced Technical Analysis")
            fig = create_advanced_charts(data, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("## 📊 Correlation Analysis")
            corr_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            fig_corr = px.imshow(corr_data, text_auto=True, aspect="auto",
                                title="Price Correlation Matrix",
                                color_continuous_scale='Viridis',
                                template='plotly_white')
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Tab 3: Model Insights
        with tab3:
            if 'ml_results' in st.session_state:
                results = st.session_state['ml_results']
                
                st.markdown("## 🤖 Model Performance Comparison")
                
                # Model metrics table
                metrics_data = []
                for model_name, result in results.items():
                    if model_name != 'features':
                        metrics_data.append({
                            'Model': model_name,
                            'MAE ($)': f"{result['mae']:.2f}",
                            'RMSE ($)': f"{result['rmse']:.2f}",
                            'R² Score': f"{result['r2']:.3f}",
                            'MAPE (%)': f"{result['mape']:.2f}",
                            'Next Day ($)': f"${result['next_day']:.2f}"
                        })
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                
                # Error heatmap
                st.markdown("## 📊 Error Metrics Heatmap")
                fig_error = create_error_heatmap(results)
                if fig_error:
                    st.plotly_chart(fig_error, use_container_width=True)
                
                # Feature importance for tree models
                if any('feature_importance' in result for result in results.values()):
                    st.markdown("## 🔑 Feature Importance")
                    col1, col2 = st.columns(2)
                    
                    # Find tree-based models
                    tree_models = [name for name, result in results.items() 
                                  if name != 'features' and 'feature_importance' in result]
                    
                    if tree_models:
                        with col1:
                            if 'Random Forest' in tree_models:
                                rf_importance = pd.DataFrame({
                                    'Feature': results['features'],
                                    'Importance': results['Random Forest']['feature_importance']
                                }).sort_values('Importance', ascending=True)
                                
                                fig_rf = px.bar(rf_importance.tail(10), x='Importance', y='Feature',
                                               title='Random Forest - Top 10 Features',
                                               orientation='h',
                                               color='Importance',
                                               color_continuous_scale='Viridis',
                                               template='plotly_white')
                                st.plotly_chart(fig_rf, use_container_width=True)
                        
                        with col2:
                            if 'Gradient Boosting' in tree_models:
                                gb_importance = pd.DataFrame({
                                    'Feature': results['features'],
                                    'Importance': results['Gradient Boosting']['feature_importance']
                                }).sort_values('Importance', ascending=True)
                                
                                fig_gb = px.bar(gb_importance.tail(10), x='Importance', y='Feature',
                                               title='Gradient Boosting - Top 10 Features',
                                               orientation='h',
                                               color='Importance',
                                               color_continuous_scale='Plasma',
                                               template='plotly_white')
                                st.plotly_chart(fig_gb, use_container_width=True)
        
        # Tab 4: Price Forecast
        with tab4:
            if 'ml_results' in st.session_state:
                results = st.session_state['ml_results']
                test_dates = st.session_state['test_dates']
                y_test = st.session_state['y_test']
                
                st.markdown("## 🔮 Price Predictions")
                
                # Prediction comparison chart
                fig_pred = create_prediction_comparison(results, test_dates, y_test)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Prediction error chart
                st.markdown("## 📉 Prediction Errors Distribution")
                fig_errors = go.Figure()
                
                for model_name, result in results.items():
                    if model_name != 'features':
                        errors = np.abs(result['predictions'] - y_test)
                        fig_errors.add_trace(go.Box(y=errors, name=model_name, 
                                                    marker_color='#1e3c72'))
                
                fig_errors.update_layout(title='Prediction Error Distribution by Model',
                                        yaxis_title='Absolute Error ($)',
                                        template='plotly_white',
                                        height=600)
                st.plotly_chart(fig_errors, use_container_width=True)
                
                # Next day predictions
                st.markdown("## 🎯 Next Day Predictions")
                model_count = len([m for m in results.keys() if m != 'features'])
                if model_count > 0:
                    next_day_cols = st.columns(min(model_count, 4))
                    
                    idx = 0
                    for model_name, result in results.items():
                        if model_name != 'features' and idx < len(next_day_cols):
                            with next_day_cols[idx]:
                                st.markdown(f"""
                                <div class="model-card">
                                    <h4>{model_name}</h4>
                                    <h2>${result['next_day']:.2f}</h2>
                                    <p>Change: {((result['next_day'] - current_price)/current_price*100):+.2f}%</p>
                                    <small>R² Score: {result['r2']:.3f}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            idx += 1
        
        # Tab 5: Risk Analytics
        with tab5:
            st.markdown("## 📊 Risk & Statistical Analysis")
            
            # Returns analysis
            returns = data['Close'].pct_change().dropna()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📈 Returns Statistics")
                st.metric("Mean Daily Return", f"{returns.mean()*100:.3f}%")
                st.metric("Std Deviation", f"{returns.std()*100:.3f}%")
                st.metric("Sharpe Ratio", f"{returns.mean()/returns.std()*np.sqrt(252):.3f}")
                st.metric("Skewness", f"{returns.skew():.3f}")
                st.metric("Kurtosis", f"{returns.kurtosis():.3f}")
            
            with col2:
                st.markdown("### ⚠️ Risk Metrics")
                st.metric("Value at Risk (95%)", f"{np.percentile(returns, 5)*100:.2f}%")
                st.metric("Conditional VaR", f"{returns[returns <= np.percentile(returns, 5)].mean()*100:.2f}%")
                st.metric("Max Drawdown", f"{(data['Close'].min()/data['Close'].max() - 1)*100:.2f}%")
                st.metric("Volatility (Annual)", f"{returns.std()*np.sqrt(252)*100:.2f}%")
                st.metric("Calmar Ratio", f"{returns.mean()*252 / (abs((data['Close'].min()/data['Close'].max() - 1))):.2f}")
            
            with col3:
                st.markdown("### 📉 Price Statistics")
                st.metric("All-time High", f"${data['Close'].max():.2f}")
                st.metric("All-time Low", f"${data['Close'].min():.2f}")
                st.metric("Average Price", f"${data['Close'].mean():.2f}")
                st.metric("Median Price", f"${data['Close'].median():.2f}")
                st.metric("Price Range", f"${data['Close'].max() - data['Close'].min():.2f}")
            
            # Trading recommendation
            if 'ml_results' in st.session_state:
                st.markdown("## 💡 Trading Recommendation")
                recommendation = generate_trading_recommendation(results, current_price)
                
                if recommendation:
                    st.markdown(f"""
                    <div class="{recommendation['class']}">
                        <h3>{recommendation['signal']} SIGNAL</h3>
                        <p><strong>Expected Price:</strong> ${recommendation['next_day_price']:.2f}</p>
                        <p><strong>Expected Change:</strong> {recommendation['expected_change']:+.2f}%</p>
                        <p><strong>Confidence Level:</strong> {recommendation['confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Tab 6: Performance Summary
        with tab6:
            st.markdown("## 🏆 Model Performance Summary")
            
            if 'ml_results' in st.session_state:
                results = st.session_state['ml_results']
                
                # Find best model
                best_model = None
                best_r2 = -float('inf')
                best_mae = float('inf')
                
                for model_name, result in results.items():
                    if model_name != 'features' and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_model = model_name
                        best_mae = result['mae']
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🏆 Best Performing Model: {best_model}</h2>
                    <h3>R² Score: {best_r2:.3f}</h3>
                    <h3>MAE: ${best_mae:.2f}</h3>
                    <p>This model explains {best_r2*100:.1f}% of the price variance</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model ranking
                st.markdown("## 📊 Model Rankings")
                rankings = []
                for model_name, result in results.items():
                    if model_name != 'features':
                        rankings.append({
                            'Model': model_name,
                            'R² Score': result['r2'],
                            'MAE ($)': result['mae'],
                            'RMSE ($)': result['rmse'],
                            'MAPE (%)': result['mape']
                        })
                
                rankings_df = pd.DataFrame(rankings)
                rankings_df = rankings_df.sort_values('R² Score', ascending=False)
                rankings_df.index = range(1, len(rankings_df) + 1)
                
                st.dataframe(rankings_df, use_container_width=True)
                
                # Recommendations
                st.markdown("## 💡 Investment Recommendations")
                
                if best_r2 > 0.8:
                    st.success("✅ **Highly Reliable Predictions** - Models show excellent performance. Consider using these predictions for investment decisions.")
                elif best_r2 > 0.6:
                    st.info("📊 **Moderately Reliable Predictions** - Models show good performance. Use predictions with proper risk management.")
                else:
                    st.warning("⚠️ **Limited Reliability** - Models show moderate performance. Consider using as a supplementary tool.")
                
                st.markdown("## 🎯 Key Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Best Features for Prediction:**")
                    if 'feature_importance' in results.get('Random Forest', {}):
                        importance_df = pd.DataFrame({
                            'Feature': results['features'],
                            'Importance': results['Random Forest']['feature_importance']
                        }).sort_values('Importance', ascending=False)
                        
                        for i, row in importance_df.head(5).iterrows():
                            st.write(f"• {row['Feature']}: {row['Importance']:.3f}")
                
                with col2:
                    st.markdown("**Model Strengths:**")
                    st.write("✅ Random Forest: Best for capturing non-linear patterns")
                    st.write("✅ Gradient Boosting: Good for complex relationships")
                    st.write("✅ Ridge Regression: Handles multicollinearity well")
                    st.write("✅ SVR: Useful for small datasets")
                
                # Risk disclaimer
                st.markdown("---")
                st.markdown("""
                <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107;">
                    <h4 style="color: #856404;">⚠️ Risk Disclaimer</h4>
                    <p style="color: #856404; margin: 0;">
                        This dashboard provides predictions based on historical data and machine learning models. 
                        Past performance does not guarantee future results. Always conduct your own research and 
                        consult with financial advisors before making investment decisions. The predictions are 
                        for educational purposes only.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>📊 StockVision AI</strong> | Intelligent Stock Predictions Powered by Machine Learning</p>
        <p style="font-size: 0.8rem;">Data Source: Yahoo Finance | Real-time Analysis | For Educational Purpose Only</p>
    </div>
    """, unsafe_allow_html=True)

# Main execution
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    # Show login page or main app
    if not st.session_state['authenticated']:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()