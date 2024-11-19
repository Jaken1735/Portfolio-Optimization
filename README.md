# Portfolio Optimization Intro

## Overview
This is an introductory project that looks at a couple of different areas for portfolio optimization, combining historical stock data, machine learning models, and optimization techniques such as the Markowitz Efficient Frontier. The primary goal is to predict stock prices and allocate assets in a portfolio to maximize returns while minimizing risk.

## Markowitz Efficient Frontier:

This part is done in order to determine the best weights for the stocks that have been picked for this analysis. 

<img width="1277" alt="Skärmavbild 2024-11-18 kl  21 01 35" src="https://github.com/user-attachments/assets/f9cadcb5-b922-4876-b419-d4b09ddea515">

---

## Monte-Carlo Simulation of Portfolio

Given the determined weights from the Markowitz approach, we can use these to simulate the development of a portfolio with these stocks. For the simulation, we use the Cholesky decomposition to transform uncorrelated samples from a normal distribution into correlated samples that match the covariance structure of the assets in the portfolio. This allows us to simulate realistic asset price paths that respect the observed correlations between the assets.

The simulation has the following parameters:

- 1000 runs
- 200 day Timeframe
- $10000 intial portfolio-value

