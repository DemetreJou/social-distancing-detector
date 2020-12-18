from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import cv2


class Heatmap:
    """
    Heatmap class used to store historical social distancing violation frame data
    Can be used to show a live view of overlayed historical data or save a single frame to file 
    """
    def __init__(self, height, width):
        self.historical = np.zeros((height, width))
        self.fig, self.ax = plt.subplots()

    @property
    def heatmap(self):
        return gaussian_filter(self.historical, 3)

    def add_frame_to_heatmap(self, frame):
        if frame.ndim == 3:
            frame = np.mean(frame, 2)

        # Non zero points (ie. violation locations) count as 1 point in the historical dataset
        frame[frame > 1] = 1

        global historical
        self.historical += frame

    def save_heatmap_plot(self, name, frame, M_inv):
        self.ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inv_warpped = cv2.warpPerspective(self.heatmap, M_inv, self.historical.shape[::-1])
        self.ax.pcolormesh(inv_warpped, cmap='seismic', alpha=0.15)
        plt.show()

        self.fig.savefig(name)
        self.ax.clear()

    def show_heatmap_plot(self, frame, M_inv):
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inv_warpped = cv2.warpPerspective(self.heatmap, M_inv, self.historical.shape[::-1])
        self.ax.pcolormesh(inv_warpped, cmap='seismic', vmin=-self.historical.max(), vmax=self.historical.max(), alpha=0.15)
        plt.pause(0.001)

