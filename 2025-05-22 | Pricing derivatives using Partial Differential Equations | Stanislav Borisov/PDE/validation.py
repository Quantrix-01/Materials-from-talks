from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def validate_call_at_a_date(S, K, T, r, sigma, V, t):
    analytical = black_scholes_call(S, K, t, r, sigma)

    validation_timepoint = int((t/T) * V.shape[1])
    numerical = V[:, validation_timepoint]

    plt.figure(figsize=(10, 6))
    plt.plot(S, analytical, label='Analytical Solution', linewidth=2)
    plt.plot(S, numerical, '--', label='Numerical Solution', linewidth=2)
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    plt.title('Comparison at t=0')
    plt.legend()
    plt.grid(True)
    plt.show()

def validate_digital_call_at_a_date(S, K, T, r, sigma, V, t):
    analytical = 1 / (2 * 0.001) * (black_scholes_call(S, K-0.001, t, r, sigma) - black_scholes_call(S, K+0.001, t, r, sigma))

    validation_timepoint = int((t/T) * V.shape[1])
    numerical = V[:, validation_timepoint]

    plt.figure(figsize=(10, 6))
    plt.plot(S, analytical, label='Analytical Solution', linewidth=2)
    plt.plot(S, numerical, '--', label='Numerical Solution', linewidth=2)
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    plt.title('Comparison at t=0')
    plt.legend()
    plt.grid(True)
    plt.show()
