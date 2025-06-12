import numpy as np
import scipy.linalg as linalg

def european_call_fd_implicit(S_max, T, K, r, sigma, M, N):
    """
    Solve Black-Scholes PDE for European call option using implicit finite difference method.
    
    Parameters:
    S_max : float
        Maximum stock price in the grid
    T : float
        Time to maturity (in years)
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    M : int
        Number of stock price steps
    N : int
        Number of time steps
    
    Returns:
    tuple : (S_values, tau_values, V)
        S_values: Array of stock prices
        tau_values: Array of time points (0 is maturity)
        V: Option value matrix
    """
    
    # Create grid
    dt = T / N
    S_values = np.linspace(0, S_max, M+1)
    tau_values = np.linspace(0, T, N+1)
    
    # Initialize option value matrix
    V = np.zeros((M+1, N+1))
    
    # Set terminal condition (payoff at maturity)
    V[:, 0] = np.maximum(S_values - K, 0)
    
    # Set boundary conditions
    V[0, :] = 0  # At S=0, call option is worthless
    V[-1, :] = S_max - K * np.exp(-r * tau_values)  # At S=S_max
    
    # Finite difference coefficients. Here: j = S_i / dS
    j = np.arange(1, M)
    alpha = 0.5 * dt * (sigma**2 * j**2 - r * j)
    beta = -dt * (sigma**2 * j**2 + r)
    gamma = 0.5 * dt * (sigma**2 * j**2 + r * j)
    
    # Create the tridiagonal matrix
    A = np.diag(1 - beta)  # Main diagonal
    A += np.diag(-alpha[1:], -1)  # Lower diagonal
    A += np.diag(-gamma[:-1], 1)  # Upper diagonal
    
    for n in range(N):
        # Right-hand side vector (including boundary conditions)
        rhs = V[1:M, n].copy()
        rhs[0] += alpha[0] * V[0, n+1]  # Boundary at S=0
        rhs[-1] += gamma[-1] * V[-1, n+1]  # Boundary at S=S_max
        
        # Solve the system
        V[1:M, n+1] = linalg.solve(A, rhs)
    
    return S_values, tau_values, V


def european_call_fd(S_max, T, K, r, sigma, M, N):
    """
    Solve Black-Scholes PDE for European call option using finite difference method.
    
    Parameters:
    S_max : float
        Maximum stock price in the grid
    T : float
        Time to maturity (in years)
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    M : int
        Number of stock price steps
    N : int
        Number of time steps
    
    Returns:
    tuple : (S_values, tau_values, V)
        S_values: Array of stock prices
        tau_values: Array of time points (in reverse, 0 is maturity)
        V: Option value matrix
    """
    
    # Create grid
    dt = T / N
    S_values = np.linspace(0, S_max, M+1)
    tau_values = np.linspace(0, T, N+1)
    
    # Initialize option value matrix
    V = np.zeros((M+1, N+1))
    
    # Set terminal condition (payoff at maturity)
    V[:, 0] = np.maximum(S_values - K, 0)
    
    # Set boundary conditions
    V[0, :] = 0  # At S=0, call option is worthless
    V[-1, :] = S_max - K * np.exp(-r * tau_values)  # At S=S_max
    
    # Finite difference coefficients
    alpha = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 - r * np.arange(M+1))
    beta = -dt * (sigma**2 * np.arange(M+1)**2 + r)
    gamma = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 + r * np.arange(M+1))
    
    for j in range(N):
        for i in range(1, M):
            V[i, j+1] = alpha[i] * V[i-1, j] + (1 + beta[i]) * V[i, j] + gamma[i] * V[i+1, j]
    
    return S_values, tau_values, V