# -*- coding: utf-8 -*-
"""
________  ___  ___  ________  ________  ___  ________  _______   ________  ___  _________  ________
|\   ____\|\  \|\  \|\   ____\|\   ____\|\  \|\   __  \|\  ___ \ |\   __  \|\  \|\___   ___\\   __  \
\ \  \___|\ \  \\\  \ \  \___|\ \  \___|\ \  \ \  \|\  \ \   __/|\ \  \|\  \ \  \|___ \  \_\ \  \|\  \
 \ \  \  __\ \  \\\  \ \  \    \ \  \    \ \  \ \   ____\ \  \_|/_\ \   ____\ \  \   \ \  \ \ \  \\\  \
  \ \  \|\  \ \  \\\  \ \  \____\ \  \____\ \  \ \  \___|\ \  \_|\ \ \  \___|\ \  \   \ \  \ \ \  \\\  \
   \ \_______\ \_______\ \_______\ \_______\ \__\ \__\    \ \_______\ \__\    \ \__\   \ \__\ \ \_______\
    \|_______|\|_______|\|_______|\|_______|\|__|\|__|     \|_______|\|__|     \|__|    \|__|  \|_______|

"""

# Installation of required libraries
!pip install pandas numpy yfinance pyportfolioopt matplotlib

# Importing libraries
import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np  # NumPy for numerical computing
import yfinance as yf  # yfinance for fetching historical stock data
import matplotlib.pyplot as plt  # Matplotlib for plotting
from pypfopt import EfficientFrontier, risk_models, expected_returns, discrete_allocation  # PyPortfolioOpt for portfolio optimization

# Definition of stock tickers and data period
tickers = ['GCG-A.TO', 'UIS', 'BFH', 'BAM', 'WED.V', 'VEL', 'PRCH', 'JXN']  # List of stock tickers
start = '2018-12-31'  # Start date of historical data
end = '2024-02-28'  # End date of historical data

# Downloading historical stock data
prices_df = yf.download(' '.join(tickers), start, end)['Adj Close']  # Fetching adjusted close prices for tickers
returns_df = prices_df.pct_change()[1:]  # Calculating daily returns for each stock

# Setting up plot dimensions
fig = plt.gcf()  # Get current figure
fig.set_size_inches(18.5, 10.5)  # Set figure size
fig.savefig('test2png.png', dpi=100)  # Save figure as PNG

# Visualizing stock/portfolio performance
title = 'Stocks Performance'  # Title of the plot
my_stocks = prices_df  # Assigning prices to my_stocks variable
for c in my_stocks.columns.values:  # Iterating over each stock
    plt.plot(my_stocks[c], label=c)  # Plotting stock prices
plt.title(title)  # Setting title of the plot
plt.xlabel('Date (Years)', fontsize=10)  # Setting label for x-axis
plt.ylabel('Price USD(Adj Close)', fontsize=10)  # Setting label for y-axis
plt.legend(my_stocks.columns.values, loc='upper left')  # Adding legend
plt.grid(axis='y')  # Adding gridlines
plt.show()  # Displaying the plot

# Calculation of annual returns and covariance matrix
r = ((1+returns_df).prod())**(252/len(returns_df)) - 1  # Calculating annualized returns
cov = returns_df.cov()*252  # Calculating covariance matrix
e = np.ones(len(r))  # Creating array of ones
mu = expected_returns.mean_historical_return(prices_df)  # Calculating historical returns
S = risk_models.sample_cov(prices_df)  # Calculating sample covariance matrix
ef = EfficientFrontier(mu, S)  # Creating EfficientFrontier object
raw_weights = ef.max_sharpe()  # Calculating raw max Sharpe ratio weights
cleaned_weights = ef.clean_weights()  # Cleaning weights for better readability
latest_prices = discrete_allocation.get_latest_prices(prices_df)  # Getting latest prices of assets

# Portfolio allocation
weights = cleaned_weights  # Using cleaned weights for allocation
da = discrete_allocation.DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)  # Creating DiscreteAllocation object
allocation, leftover = da.greedy_portfolio()  # Allocating portfolio and calculating leftover funds

# Defining variables for tangent portfolio calculation
icov = np.linalg.inv(cov)  # Calculating inverse of covariance matrix
h = np.matmul(e, icov)  # Calculating h vector
g = np.matmul(r, icov)  # Calculating g vector
a = np.sum(e*h)  # Calculating sum of e*h
b = np.sum(r*h)  # Calculating sum of r*h
c = np.sum(r*g)  # Calculating sum of r*g
d = a*c - b**2  # Calculating determinant of covariance matrix

# Calculation of minimum tangency portfolio and its variance
mvp = h/a  # Calculating minimum variance portfolio
mvp_returns = b/a  # Calculating expected return of minimum variance portfolio
mvp_risk = (1/a)**(1/2)  # Calculating standard deviation of minimum variance portfolio
tagency = g/b  # Calculating tangent portfolio
tagency_returns = c/b  # Calculating expected return of tangent portfolio
tagency_risk = c**(1/2)/b  # Calculating standard deviation of tangent portfolio

# Plotting efficient frontier
exp_returns = np.arange(-0.001, 0.5, 0.001)  # Generating range of expected returns
risk = ((a*exp_returns**2 - 2*b*exp_returns + c)/d)**(1/2)  # Calculating risk for each expected return

plt.plot(risk, exp_returns, linestyle='dotted', color='b')  # Plotting efficient frontier
plt.scatter(mvp_risk, mvp_returns, marker='*', color='r')  # Marking minimum variance portfolio
plt.scatter(tagency_risk, tagency_returns, marker='*', color='g')  # Marking tangent portfolio
plt.title("Efficient Frontier")  # Setting title of the plot
plt.xlabel("Standard Deviation")  # Setting label for x-axis
plt.ylabel("Expected Return")  # Setting label for y-axis
plt.grid(axis='y')  # Adding gridlines
plt.legend(["Efficient Frontier", "Efficient portfolio with minimum volatility", "Optimal risky portfolio"], loc="lower right")  # Adding legend

# Plotting securities market line
SML_slope = 1/c**(1/2)  # Calculating slope of securities market line
SML_risk = exp_returns*SML_slope  # Calculating risk for each expected return on securities market line
plt.plot(risk, exp_returns, color='b', linestyle='dotted')  # Plotting efficient frontier
plt.plot(SML_risk, exp_returns, color='r', linestyle='dashdot')  # Plotting securities market line
plt.scatter(mvp_risk, mvp_returns, marker='*', color='r')  # Marking minimum variance portfolio
plt.scatter(tagency_risk, tagency_returns, marker='*', color='g')  # Marking tangent portfolio
plt.title("Efficient Frontier & Securities Market Line")  # Setting title of the plot
plt.xlabel("Standard Deviation")  # Setting label for x-axis
plt.ylabel("Expected Return")  # Setting label for y-axis
plt.grid(axis='y')  # Adding gridlines
plt.legend(["Efficient Frontier", "Securities Market Line (SML)", "Efficient portfolio with minimum volatility", "Optimal risky portfolio"], loc="lower right")  # Adding legend

# Solving the target return problem
target_return = 0.2667869983529349  # Specifying target return
if target_return < mvp_returns:  # Checking if target return is less than minimum variance portfolio return
  optimal_portfolio = mvp  # Assigning minimum variance portfolio as optimal portfolio
  optimal_return = mvp_returns  # Assigning minimum variance portfolio return as optimal return
  optimal_risk = mvp_risk  # Assigning minimum variance portfolio risk as optimal risk
else:  # If target return is greater than minimum variance portfolio return
  l = (c - b*target_return)/d  # Calculating l factor
  m = (a * target_return - b)/d  # Calculating m factor
  optimal_portfolio = l*h + m*g  # Calculating optimal portfolio
  optimal_return = np.sum(optimal_portfolio*r)  # Calculating optimal return
  optimal_risk = ((a*optimal_return**2 - 2*b*optimal_return + c)/d)**(1/2)  # Calculating optimal risk

print(optimal_portfolio, optimal_return, optimal_risk)  # Printing optimal portfolio, return, and risk

ef.portfolio_performance(verbose=True)  # Displaying portfolio performance metrics

print("Cleaned Weights:", cleaned_weights)  # Printing cleaned weights
print("Discrete Allocation:", allocation)  # Printing discrete allocation
print("Remaining Funds: ${:.2f}".format(leftover), "CAD")  # Printing remaining funds
