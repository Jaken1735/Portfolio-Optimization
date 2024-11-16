# Portfolio Optimization

## Overview
This project implements a complete pipeline for portfolio optimization, leveraging historical stock data, machine learning models, and optimization techniques. The main goal is to predict stock prices and allocate assets in a portfolio to maximize returns while minimizing risk.

### Key Components:
1. **Data Processing:**
   - Importing data from S&P 500 stocks on yFinance, see data.py
   - Loading historical stock data from the created CSV files.
   - Feature engineering to calculate technical indicators such as moving averages, RSI, Bollinger Bands, and momentum.

2. **Machine Learning:**
   - A Long Short-Term Memory (LSTM) model implemented in PyTorch to predict future stock prices.

3. **Portfolio Optimization:**
   - Use optimization techniques (e.g., maximizing the Sharpe ratio) to determine optimal stock allocations based on predictions.

---

## Features

### 1. Data Loading and Feature Engineering
- Loads multiple CSV files into Pandas DataFrames.
- Calculates technical indicators for each stock to enrich the dataset.
- Outputs processed datasets for further analysis or machine learning.

### 2. Machine Learning (LSTM Model)
- Predicts future stock prices using an LSTM neural network.
- Supports configurable parameters such as:
  - Sequence length
  - Hidden layers
  - Learning rate
  - Number of epochs
- Utilizes GPU acceleration if available.

### 3. Portfolio Optimization
- Future stock prices are fed into an optimization model.
- Allocates weights to stocks to achieve desired objectives:
  - Maximizing returns
  - Minimizing volatility
  - Maximizing the Sharpe ratio.

---

## Installation

