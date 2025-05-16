LSTM Stock Price Forecasting App
================================

This is a Streamlit web application for forecasting the closing prices of multiple stocks using LSTM (Long Short-Term Memory) neural networks. The app fetches historical data from Yahoo Finance and visualizes both training and prediction results.

Features
--------
- Select one or multiple stock symbols.
- Download historical data using yFinance.
- Train LSTM models on selected stocks.
- Predict future closing prices.
- Interactive graphs and user interface.

Technologies Used
-----------------
- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- yFinance
- Scikit-learn

Installation
------------
1. Clone the repository:
   git clone https://github.com/yourusername/lstm-stock-forecast.git
   cd lstm-stock-forecast

2. (Optional) Create a virtual environment:
   python -m venv venv
   On Linux/macOS: source venv/bin/activate
   On Windows:     venv\Scripts\activate

3. Install the required Python packages:
   pip install -r requirements.txt

4. Run the Streamlit app:
   streamlit run app.py

Usage
-----
1. Launch the app in your browser after starting it.
2. Select the stock symbols you're interested in.
3. Click "Fetch & Predict" to load data and train models.
4. View training and forecasted price charts for each stock.

Model Info
----------
- LSTM neural networks are used for time series forecasting.
- A sliding window of 60 days is used for input sequences.
- Data is normalized using MinMaxScaler before training.

File Structure
--------------
- app.py              -> Main Streamlit application
- requirements.txt    -> Required Python libraries
- README.txt          -> Project overview and setup guide

Notes
-----
- Requires an internet connection to fetch stock data.
- Training multiple models may take a few minutes depending on system performance.

License
-------
This project is licensed under the MIT License.
