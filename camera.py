import cv2
import threading
import time

class Camera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.is_running = True
        self.frame = None
        self.lock = threading.Lock()
        
        # Start the thread to read frames
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.is_running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.capture.release()
