import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


class Spectrogram:
    def __init__(self, fs, baseline=2, nperseg=256, noverlap=255):
        self.fs = fs
        self.baseline = baseline
        self.nperseg = nperseg
        self.noverlap = noverlap

    def compute_spectrogram(self, data):
        """
        data: numpy array o kształcie (n_trials, n_channels, n_samples)
        Zwraca: Sxx - shape (n_trials, n_channels, n_freqs, n_times)
        """
        n_trials, n_channels, _ = data.shape
        spec_list = []

        for trial in range(n_trials):
            trial_spec = []
            for ch in range(n_channels):
                f, t, Sxx = spectrogram(data[trial, ch, :], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
                trial_spec.append(Sxx)
            spec_list.append(trial_spec)

        Sxx = np.array(spec_list)  # shape: (n_trials, n_channels, n_freqs, n_times)
        t = t - self.baseline
        t = t - (self.nperseg / 2) / self.fs
        return f, t, Sxx

    def transform(self, data, f_range=(8, 30), t_range=(0, 5)):
        freqs, times, Sxx = self.compute_spectrogram(data)
        f_idx = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[0]
        t_idx = np.where((times >= t_range[0]) & (times <= t_range[1]))[0]
        Sxx = Sxx[:, :, f_idx[:, None], t_idx]
        return Sxx

    def normalize_mean(self, Sxx):
        """
        Normalizacja przez odejmowanie średniej z baseline i dzielenie przez baseline.
        """
        baseline_samples = int(self.baseline * self.fs / (self.nperseg - self.noverlap))
        baseline_mean = Sxx[:, :, :, :baseline_samples].mean(axis=(0, 3), keepdims=True)
        Sxx_mean = Sxx.mean(axis=0, keepdims=True)  # uśrednianie po trialach
        Sxx_norm = (Sxx_mean - baseline_mean) / baseline_mean
        return Sxx_norm.squeeze()

    def normalize_zscore(self, Sxx):
        """
        Normalizacja z-score: (X - mean) / std na podstawie baseline.
        """
        baseline_samples = int(self.baseline * self.fs / (self.nperseg - self.noverlap))
        baseline_mean = Sxx[:, :, :, :baseline_samples].mean(axis=(0, 3), keepdims=True)
        baseline_std = Sxx[:, :, :, :baseline_samples].std(axis=(0, 3), keepdims=True)
        Sxx_mean = Sxx.mean(axis=0, keepdims=True) # uśrednianie po trialach
        Sxx_norm = (Sxx_mean - baseline_mean) / baseline_std
        return Sxx_norm.squeeze()

    def plot_head(self, Sxx, freqs, times, names, f_range=(8, 30), t_range=(0, 5), title=None):
        f_idx = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[0]
        t_idx = np.where((times >= t_range[0]) & (times <= t_range[1]))[0]
        data = Sxx[:, f_idx[:, None], t_idx]

        topology_map = {
            "Fp1": (0, 1), "Fpz": (0, 2), "Fp2": (0, 3),
            "F7": (1, 0), "F3": (1, 1), "Fz": (1, 2), "F4": (1, 3), "F8": (1, 4),
            "T3": (2, 0), "C3": (2, 1), "Cz": (2, 2), "C4": (2, 3), "T4": (2, 4),
            "T5": (3, 0), "P3": (3, 1), "Pz": (3, 2), "P4": (3, 3), "T6": (3, 4),
            "O1": (4, 1), "O2": (4, 3)
        }

        fig, axs = plt.subplots(5, 5, figsize=(12, 10))
        fig.subplots_adjust(top=0.92)

        for i, ch in enumerate(names):
            if ch not in topology_map:
                continue
            row, col = topology_map[ch]
            im = axs[row, col].imshow(
                data[i],
                aspect='auto',
                origin='lower',
                extent=(times[t_idx[0]], times[t_idx[-1]], freqs[f_idx[0]], freqs[f_idx[-1]]),
                interpolation='nearest',
                cmap='seismic'
            )
            axs[row, col].set_title(ch, fontsize=12)
            axs[row, col].set_xlabel('Time (s)')
            axs[row, col].set_ylabel('Freq (Hz)')

        for ax in axs.flatten():
            if not ax.has_data():
                ax.axis('off')

        if title:
            plt.suptitle(title, fontsize=20)

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    def plot_components(self, Sxx, freqs, times, f_range=(8, 30), t_range=(0, 5), title=None):
        f_idx = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[0]
        t_idx = np.where((times >= t_range[0]) & (times <= t_range[1]))[0]
        data = Sxx[:, f_idx[:, None], t_idx]

        n_ch = data.shape[0]

        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        axs = axs.flatten()

        for i in range(n_ch):
            im = axs[i].imshow(
                data[i],
                aspect='auto',
                origin='lower',
                extent=(t_range[0], t_range[1], f_range[0], f_range[1]),
                interpolation='nearest',
                cmap='seismic'
            )

            axs[i].set_title(f'Channel {i + 1}')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Freq (Hz)')

        for j in range(n_ch, len(axs)):
            axs[j].axis('off')

        if title:
            plt.suptitle(title)

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()