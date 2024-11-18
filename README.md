# Portfolio Optimization

## Overview
This project implements a comprehensive pipeline for portfolio optimization, combining historical stock data, machine learning models, and optimization techniques such as the Markowitz Efficient Frontier. The primary goal is to predict stock prices and allocate assets in a portfolio to maximize returns while minimizing risk.

### Key Components:
1. **Data Processing:**
   - Importing data from S&P 500 stocks on yFinance, see data.py
   - Loading historical stock data from the created CSV files.
   - Feature engineering to calculate technical indicators such as moving averages, RSI, Bollinger Bands, and momentum.

2. **Machine Learning:**
   - A Long Short-Term Memory (LSTM) model implemented in PyTorch to predict future stock prices.

3. **Portfolio Optimization:**
   - Use optimization techniques (e.g., maximizing the Sharpe ratio) to determine optimal stock allocations based on predictions.

4. **Markowitz Efficient Frontier:**
   - Risk-Return Analysis: Computes portfolio returns, risks (volatility), and covariances to identify the optimal risk-return tradeoff.
   - Visualization: Plots the Efficient Frontier and highlights the optimal portfolio.

<img width="1277" alt="Skärmavbild 2024-11-18 kl  21 01 35" src="https://github.com/user-attachments/assets/f9cadcb5-b922-4876-b419-d4b09ddea515">

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

