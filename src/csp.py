import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class CSP:
    def __init__(self):
        self.W = None
        self.Lambda = None

    def fit(self, L, R):
        n_rep = L.shape[0]
        n_ch = L.shape[1]

        R_L = np.zeros((n_ch, n_ch))
        R_R = np.zeros((n_ch, n_ch))
        for r in range(n_rep):
            L_temp = L[r].T
            tmp = np.cov(L_temp, rowvar=False)
            regularize = 1e-5 * np.eye(tmp.shape[0])
            tmp += regularize
            R_L += tmp / np.trace(tmp)

            R_temp = R[r].T
            tmp = np.cov(R_temp, rowvar=False)
            regularize = 1e-5 * np.eye(tmp.shape[0])
            tmp += regularize
            R_R += tmp / np.trace(tmp)

        R_L /= n_rep
        R_R /= n_rep

        # Rozwiązanie uogólnionego zagadnienia własnego
        Lambda, W = eigh(R_L, R_R)

        self.Lambda = Lambda
        self.W = W

    def transform(self, data):
        if self.W is None:
            raise ValueError("Model CSP nie został wytrenowany. Najpierw wywołaj metodę fit().")

        S = np.zeros_like(data)
        for r in range(data.shape[0]):
            S[r, :, :] = self.W.T @ data[r]
        return S

    def fit_transform(self, L, R):
        self.fit(L, R)

        L_trans = self.transform(L)
        R_trans = self.transform(R)

        return L_trans, R_trans

    def plot_component(self, n_component, names, title=None):
        if self.W is None:
            raise ValueError("Musisz najpierw wywołać metodę fit(), aby wyznaczyć W.")

        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        fig.subplots_adjust(top=0.92)

        if title is not None:
            fig.suptitle(title, fontsize=20)

        topology_map = {
            "Fp1": (0, 1), "Fpz": (0, 2), "Fp2": (0, 3),
            "F7": (1, 0), "F3": (1, 1), "Fz": (1, 2), "F4": (1, 3), "F8": (1, 4),
            "T3": (2, 0), "C3": (2, 1), "Cz": (2, 2), "C4": (2, 3), "T4": (2, 4),
            "T5": (3, 0), "P3": (3, 1), "Pz": (3, 2), "P4": (3, 3), "T6": (3, 4),
            "O1": (4, 1), "O2": (4, 3),
        }

        W_inv = np.linalg.inv(self.W)
        W_inv_ch = W_inv[n_component, :]

        for x in range(5):
            for y in range(5):
                axs[x, y].axis('off')

        for i_ch, ch in enumerate(names):
            if ch in topology_map:
                x, y = topology_map[ch]
                square = np.full((10, 10), np.abs(W_inv_ch[i_ch]))
                im = axs[x, y].imshow(square, cmap='Reds',
                                      vmin=np.min(np.abs(W_inv_ch)),
                                      vmax=np.max(np.abs(W_inv_ch)))
                axs[x, y].set_title(ch, fontsize=12)

        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.05)
        plt.show()
