from threading import Thread
import cv2

class VideoScreenshot(object):
    def __init__(self, src=0):
        self.status = False
        self.frame = None
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Start the thread to read frames from the video stream
        # This continously calls update() to prevent the Videocapture object from holding old frames on a live webcam
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def read(self):
        # Returns the most recent frame read
        return self.status, self.frame