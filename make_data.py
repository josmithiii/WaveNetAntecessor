import numpy as np
import pandas as pd

# Generate 1000 time points
n_points = 1000
time = np.linspace(0, 10, n_points)  # 10 cycles over 1000 points
signal = np.sin(2 * np.pi * time)  # Generate sine wave

# Create class labels (0 for positive, 1 for negative)
open_channels = (signal < 0).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'time': time,
    'signal': signal,
    'open_channels': open_channels
})

# Save to CSV
df.to_csv('train_clean_kalman.csv', index=False)

# Preview first few rows
print(df.head())
