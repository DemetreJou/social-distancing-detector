from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import cv2

class Heatmap:
    def __init__(self, height, width):
        self.historical = np.zeros((height, width))
        self.fig, self.ax = plt.subplots()

    @property
    def heatmap(self):
        return gaussian_filter(self.historical, 3)

    def add_frame_to_heatmap(self, frame):
        if frame.ndim == 3:
            frame = np.mean(frame, 2)

        frame[frame > 1] = 1

        global historical
        self.historical += frame

    # name = f"./{output_folder}/colormap_frame_{frame_number}.jpg"
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

