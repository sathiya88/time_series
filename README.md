📈 Time Series Stock Forecasting Dashboard
A comprehensive web-based forecasting tool built using Streamlit, this project predicts stock prices using both statistical and deep learning models. It enables users to analyze and compare the performance of:

ARIMA

SARIMA

Prophet (by Meta)

LSTM (Neural Networks)

The interactive dashboard offers powerful visualizations and detailed evaluation metrics to help users make data-driven decisions.

🎯 Project Objective
To evaluate and compare different time series forecasting models across:

Predictive accuracy

Interpretability

Performance on recent stock market data

The project highlights strengths and trade-offs between classical time series models and modern machine learning approaches.

📊 Forecasting Models
Model	Description
ARIMA	A classic autoregressive model suitable for non-seasonal data
SARIMA	ARIMA extended with seasonal components
Prophet	Robust to outliers, built for business forecasting (by Meta)
LSTM	A deep learning model capable of learning complex temporal patterns

📏 Evaluation Metrics
Each model is validated using the last 30 days of real-world stock price data. Metrics include:

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

MAPE – Mean Absolute Percentage Error

R² Score – Variance Explained

SMAPE – Symmetric MAPE

MSLE – Mean Squared Log Error

Residual Analysis & Ljung–Box Test for autocorrelation

📌 Key Insights
🔍 LSTM produced the highest accuracy but required more computation and training time

⚖️ Prophet offered a great balance of performance and transparency

📉 ARIMA/SARIMA were efficient but less effective in handling volatile or irregular stock movements

🛠️ Tech Stack
Language: Python 3.x

Dashboard: Streamlit

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn, plotly

Modeling: statsmodels, Prophet, TensorFlow/Keras

Data Source: yfinance (Yahoo Finance API)

🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/stock-forecasting-project.git
cd stock-forecasting-project
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Launch the Streamlit App
bash
Copy
Edit
streamlit run app.py
📂 Project Structure
bash
Copy
Edit
stock-forecasting-project/
├── app.py              # Main Streamlit interface
├── utils.py            # Metrics & helper functions
├── notebook.ipynb      # Jupyter notebook for development & testing
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
🌐 Live Deployment
Deploy this app using Streamlit Cloud in minutes:

Upload this repository to a public GitHub repository

Visit streamlit.io/cloud

Link your GitHub and select app.py as the entry point

Click Deploy

Share your unique URL ✅
