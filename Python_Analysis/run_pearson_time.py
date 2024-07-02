from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

##functions
def compute_3d_correlation_matrix(signal, win, Fs, overlap):
    """
    Compute the 3D correlation matrix for each pair of signals within each overlapping window.

    Parameters:
    - signal: Input time series data as a 2D array (n_signals x time_points).
    - win: Window size in seconds.
    - Fs: Sampling frequency in Hz.
    - overlap: overlap percentage basde on window size. 1 = full step, 0.5 =  50% overlap
    Returns:
    - 3D correlation matrix (n_signals x n_signals x n_windows).
    """
    n_signals, n_time_points = signal.shape
    step_size = int(overlap * win * Fs)  # Convert step size to number of data points
    n_windows = int(np.floor((n_time_points - win * Fs) / step_size) + 1)

    correlation_matrix = np.zeros((n_signals, n_signals, n_windows))

    for window_start in range(0, n_time_points - win * Fs + 1, step_size):
        window_end = window_start + win * Fs
        for i in range(n_signals):
            for j in range(i, n_signals):
                corr, _ = pearsonr(signal[i, window_start:window_end], signal[j, window_start:window_end])
                correlation_matrix[i, j, window_start // step_size] = corr
                if i != j:
                    correlation_matrix[j, i, window_start // step_size] = corr  # Symmetric matrix

    return correlation_matrix


def plot_vectorized_correlation(correlation_matrix):
    """
    Vectorize the 3D correlation matrix and plot the 2D correlation across time.

    Parameters:
    - correlation_matrix: 3D array (n_signals x n_signals x n_windows) from compute_3d_correlation_matrix.
    """
    n_signals, _, n_windows = correlation_matrix.shape
    vectorized_matrix = correlation_matrix.reshape(n_signals ** 2, n_windows)

    plt.figure(figsize=(10, 6))
    plt.imshow(vectorized_matrix, aspect='auto', origin='lower', extent=[0, n_windows, 0, n_signals ** 2])
    plt.colorbar(label='Pearson Correlation')
    plt.xlabel('Time (window index)')
    plt.ylabel('Signal Pair Index')
    plt.title('Vectorized Correlation Across Time')
    plt.show()

#run xample

# Generate an example time series of 10 signals, each 3600 seconds long
# Base signal: random noise
time_series = np.random.rand(10, 3600)
time_series[1, :600] = time_series[0, :600] * 0.9 + np.random.rand(600) * 0.1
time_series[3, 1000:1600] = time_series[2, 1000:1600] * 0.8 + np.random.rand(600) * 0.2
time_series[5, -600:] = time_series[4, -600:] * 0.95 + np.random.rand(600) * 0.05
# plot
corr_matrix_3d = compute_3d_correlation_matrix(time_series, win=3, Fs=1, overlap =0.5)
plot_vectorized_correlation(corr_matrix_3d)
plt.show()
print('Done')
