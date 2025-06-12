import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def static_visualization(S, tau, V):
    # Plot the option value surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    S_grid, tau_grid = np.meshgrid(S, tau)
    ax.plot_surface(S_grid, tau_grid, V.T, cmap='viridis')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Value')
    ax.set_title('European Call Option Value')
    plt.show()

def interactive_visualization(S, tau, V):
    # Create meshgrid for 3D plot
    S_grid, tau_grid = np.meshgrid(S, tau)

    # Create interactive 3D surface plot with Plotly
    fig = go.Figure(data=[go.Surface(
        z=V.T,
        x=S_grid,
        y=tau_grid,
        colorscale='Viridis',
        hovertemplate=(
            "Stock Price: %{x:.2f}<br>"
            "Time to Maturity: %{y:.2f}<br>"
            "Option Value: %{z:.2f}<extra></extra>"
        )
    )])

    # Update layout for better visualization
    fig.update_layout(
        title='Derivative Value Surface',
        scene=dict(
            xaxis_title='Stock Price',
            yaxis_title='Time to Maturity',
            zaxis_title='Option Value',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)  # Adjust camera view
            )
        ),
        autosize=False,
        width=900,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90)
    )


    fig.show()