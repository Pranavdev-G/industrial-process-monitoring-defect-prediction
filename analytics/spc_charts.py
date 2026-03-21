import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_control_chart(data, title="Control Chart"):
    """
    Create Shewhart control chart.
    Returns fig, ax
    """
    if len(data) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for control chart", ha='center', va='center')
        return fig, ax

    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)

    # Control limits: ±3σ
    ucl = mean_val + 3 * std_val
    lcl = mean_val - 3 * std_val

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data, marker='o', linestyle='-', color='blue', label='Data')
    ax.axhline(mean_val, color='green', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axhline(ucl, color='red', linestyle='--', label=f'UCL: {ucl:.2f}')
    ax.axhline(lcl, color='red', linestyle='--', label=f'LCL: {lcl:.2f}')

    # Highlight outliers
    outliers = (data > ucl) | (data < lcl)
    if outliers.any():
        ax.scatter(np.where(outliers)[0], data[outliers], color='red', s=50, label='Outliers')

    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
