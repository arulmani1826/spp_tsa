import streamlit as st
st.set_page_config(layout="wide")  # Must be first Streamlit command
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# OPTIONAL: clear old models with bad config (not removing date-stamped files)
for file in os.listdir():
    if file.endswith('_lstm_model.h5') and len(file.split('_')) == 3:  # only old models without date
        os.remove(file)

# Simple Loading Screen
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown(
        """
        <div style='
            display: flex;
            height: 100vh;
            justify-content: center;
            align-items: center;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            background-color: #262730;
        '>
            ⏳ Loading Multi-Stock Forecast App...
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(2)

loading_placeholder.empty()

# App Title
st.title('📈 Multi-Stock Price Forecast with LSTM')

# Sidebar Configuration
st.sidebar.header('🛠️ Configuration')

# Company to Ticker dictionary (expandable)
company_to_ticker = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Netflix": "NFLX",
    "Intel": "INTC",
    "IBM": "IBM",
    "Adobe": "ADBE",
    "PayPal": "PYPL",
    "Oracle": "ORCL",
    "Qualcomm": "QCOM",
    "Cisco": "CSCO",
    "Broadcom": "AVGO",
    "AMD": "AMD",
    "Micron": "MU",
    "Uber": "UBER",
    "Lyft": "LYFT",
    "Spotify": "SPOT",
    "Zoom": "ZM",
    "Shopify": "SHOP",
    "Salesforce": "CRM",
    "Snowflake": "SNOW",
    "Palantir": "PLTR",
    "AT&T": "T",
    "Verizon": "VZ",
    "Boeing": "BA",
    "Lockheed Martin": "LMT",
    "Northrop Grumman": "NOC",
    "Ford": "F",
    "General Motors": "GM",
    "Walmart": "WMT",
    "Target": "TGT",
    "Coca-Cola": "KO",
    "Pepsi": "PEP",
    "McDonald's": "MCD",
    "Starbucks": "SBUX",
    "Costco": "COST",
    "Nike": "NKE",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "Moderna": "MRNA",
    "UnitedHealth Group": "UNH",
    "Procter & Gamble": "PG",
    "Visa": "V",
    "Mastercard": "MA",
    "Bank of America": "BAC",
    "JPMorgan Chase": "JPM",
    "Goldman Sachs": "GS",
    "American Express": "AXP",
    "BlackRock": "BLK",
    "Berkshire Hathaway": "BRK-B",
}

# Create lowercase mapping
company_to_ticker_lower = {k.lower(): v for k, v in company_to_ticker.items()}

# Input: company names or tickers
tickers_input = st.sidebar.text_input('Enter stock tickers or company names (comma separated)', '')
raw_inputs = [item.strip().lower() for item in tickers_input.split(',') if item.strip()]
tickers = []


for name in raw_inputs:
    if name.upper() in company_to_ticker.values():
        tickers.append(name.upper())  # Already a valid ticker
    elif name in company_to_ticker_lower:
        resolved = company_to_ticker_lower[name]
        st.sidebar.write(f"🔎 '{name}' resolved to: `{resolved}`")
        tickers.append(resolved)
    else:
        st.sidebar.warning(f"⚠️ Could not resolve '{name}' to a valid ticker.")
train_model = st.sidebar.checkbox('Train New Model', value=True)
start_date = st.sidebar.date_input('Start Date', datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.date.today())
look_back = st.sidebar.slider("Look-back Period", 30, 120, 60, step=5)
epochs = st.sidebar.slider("Epochs", 1, 100, 50, step=1)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, step=16)
forecast_days = st.sidebar.slider("Forecast Period (days)", 1, 30, 7)


if not tickers:
    st.warning("Please enter at least one valid stock ticker or company name in the sidebar.")
    st.stop()

@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise ValueError(f"Data for {ticker} is unavailable.")
        df.fillna(method='ffill', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

def get_company_name(ticker):
    try:
        company_info = yf.Ticker(ticker).info
        return company_info.get('longName', 'Company name not found')
    except Exception as e:
        return 'Company name not found'

# Store logs for training loss and metrics
train_logs = {}

def plot_prediction_chart(dates, actual, predicted, title="Stock Price Prediction"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted', line=dict(color='orange')))
    fig.update_layout(
        title=title,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=500
    )
    return fig

# Today's date string for model versioning
today_str = datetime.date.today().isoformat()

# Loop through tickers and process
for ticker in tickers:
    st.header(f'🔍 {ticker} - {get_company_name(ticker)} Analysis')
    df = load_data(ticker, start_date, end_date)

    if df.empty or len(df) < (look_back + 1):
        st.warning(f'⛔ Not enough data for {ticker}. Skipping.')
        continue

    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Model filename with today's date appended
    model_name = f'{ticker}_{today_str}_lstm_model.h5'

    with st.spinner(f"Training {ticker} model..."):
        # Load existing model if found for today and training is not forced
        if os.path.exists(model_name) and not train_model:
            model = load_model(model_name)
            st.success(f"{ticker} model loaded from cache (saved on {today_str}).")
        else:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            progress_bar = st.progress(0)
            epoch_losses = []
            for epoch in range(epochs):
                history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
                epoch_losses.append(history.history['loss'][0])
                progress_bar.progress((epoch + 1) / epochs)

            model.save(model_name)
            train_logs[ticker] = epoch_losses
            st.success(f"{ticker} model trained and saved for today ({today_str}).")

    # Evaluate model performance
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))

    st.subheader(f"{ticker} - Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    dates = df.index[look_back + train_size:]
    fig = plot_prediction_chart(dates, y_test_true.flatten(), y_pred.flatten(), f'{ticker} Price: Actual vs Predicted')
    st.plotly_chart(fig, use_container_width=True)

    # Predict future prices
    input_seq = scaled[-look_back:].reshape(1, look_back, 1)
    future_scaled = []
    for _ in range(forecast_days):
        next_val = model.predict(input_seq)[0][0]
        future_scaled.append(next_val)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)

    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    future_dates = []
    d = df.index[-1] + datetime.timedelta(days=1)
    while len(future_dates) < forecast_days:
        if d.weekday() < 5:  # Skip weekends
            future_dates.append(d)
        d += datetime.timedelta(days=1)

    forecast_df = pd.DataFrame({
        'Date': [d.date() for d in future_dates],
        f'{ticker} Forecast ($)': [f'{p:.2f}' for p in future]
    })

    st.subheader(f'📅 {ticker} - Next {forecast_days} Trading Days')
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f'{ticker}_forecast.csv',
        mime='text/csv',
    )

    last_price = close_prices[-1][0]
    forecast_last = future[-1]
    percent_change = (forecast_last - last_price) / last_price * 100
    st.metric(f'{ticker} Change Estimate', f'{percent_change:.2f}%', delta=f'{percent_change:.2f}%')
