from scipy.ndimage import gaussian_filter
import numpy


class Heatmap:
    def __init__(self, height, width):
        self.historical = np.zeros((height, width, 1))

    @property
    def heatmap(self):
        return gaussian_filter(np.sum(self.historical, 2), 3)

    def add_frame_to_heatmap(self, frame):
        if frame.ndim == 3:
            frame = np.mean(frame, 2)

        frame[frame==grey_val] = 0 # Remove the grey background
        frame[frame > 0] = 1
        frame = frame[::-1,].reshape(frame.shape + (1,))

        global historical
        self.historical = np.concatenate((self.historical, frame), axis=2)

    # name = f"./{output_folder}/colormap_frame_{frame_number}.jpg"
    def save_heatmap_plot(self, name):
        fig, ax = plt.subplots()
        ax.pcolormesh(self.heatmap, cmap='seismic', vmin=-historical_img.max(), vmax=historical_img.max(), alpha=0.5)
        fig.savefig(name)
        ax.clear()

    def show_heatmap(self):
        cv2.imshow("Heatmap", self.heatmap)