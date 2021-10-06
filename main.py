import sys
import time
import cv2
import mss
import numpy as np
import torch
from queue import Queue
from threading import Thread

mss.factory.platform.architecture()

DEVICE = 'cuda'
DESIRED_FPS = 20
DESIRED_SCREEN = {"top": 0, "left": 0, "width": 1920, "height": 1080}
DESIRED_INFERENCE_SIZE = (1280, 720)

if DEVICE == 'cuda':
    assert torch.cuda.is_available()

class ScreenShotter(Thread):
    def __init__(self, screenshot_queue:Queue):
        Thread.__init__(self,daemon=True)
        self.images = []
        self.stopped = False
        self.screen = DESIRED_SCREEN
        self.screenshot_queue = screenshot_queue

    def run(self):
        with mss.mss() as sct:
            while not self.stopped:
                if self.screenshot_queue.full():
                    time.sleep(0.03)
                    continue
                # Grab image from the screen using the values defined by self.screen
                img_np = np.array(sct.grab(self.screen), dtype=np.uint8)
                # Resize the image and put it into the Screenshot queue.
                img_np = cv2.resize(img_np, (1280, 720),interpolation=cv2.INTER_LINEAR) # resize
                self.screenshot_queue.put(img_np,block=True,timeout=5)


class ImgInference(Thread):
    def __init__(self, screenshot_queue:Queue, display_queue:Queue):
        Thread.__init__(self,daemon=True)
        self.done = False
        self.display_queue = display_queue
        self.screenshot_queue =screenshot_queue
        self.names = []
        self.stopped = False
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5m6").half().cuda()
        self.done = True

    def run(self):
        while not self.stopped:
            # Get next image to infer from the ScreenShotter thread in the background.
            img = self.screenshot_queue.get(block=True,timeout=5) # read last screen image

            # Get label names for predictions if loop has never run before.
            if len(self.names) == 0:
                dat = self.model(img)
                self.names = dat.names
                dat = dat.xyxy[0].int().cpu().numpy()
            else:
                # If label names exist, make prediction slightly more efficiently.
                dat = self.model(img).xyxy[0].int().cpu().numpy()

            # Draw predicted bounding boxes and labels onto image.
            for row in dat:
                name = self.names[row[-1]]
                xmin, ymin, xmax , ymax = row[:4]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (90, 10, 255), 1)
                cv2.putText(img, name[:10], (xmin+4, ymin+10), cv2.QT_FONT_NORMAL, 0.5, (255,255,90),thickness=1)
            
            # Send back the prepared image to display in the main thread.
            self.display_queue.put(img, block=True, timeout=5)

            # Re-Queue the ScreenShotter to get another image from the screen.
            self.screenshot_queue.task_done()


def run():


    screenshot_queue = Queue(maxsize=3)
    display_queue = Queue(maxsize=3)
    screenShotter = ScreenShotter(screenshot_queue)
    inference = ImgInference(screenshot_queue, display_queue)
    while not inference.done:
        time.sleep(0.01)
    frames = 0
    inference.start()
    screenShotter.start()
    while True:
        # Get the most recently prepared image from the display_queue.
        img = display_queue.get(block=True,timeout=5)
        # If this is the first ieration, create the fps counter reference time.
        if frames == 0:
            last_time = time.time()
        frames = frames + 1
        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", img)

        # Every 15 frames, recalculate the fps.
        if frames % 15 == 0:
            #Cleaar screen
            sys.stdout.write("\033[2J\033[1;1H")
            sys.stdout.flush()

            # Measure FPS, show queue sizes for debugging purposes
            sys.stdout.write(f"fps: {frames/(time.time() - last_time)}, dqsize: {display_queue.qsize()}, qsize:{screenshot_queue.qsize()}\n")
            sys.stdout.flush()

        # Tell inference thread to get another frame.
        display_queue.task_done()

        # Press "q" to quit.
        if cv2.waitKey(4) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            screenShotter.stopped = True
            inference.stopped = True
            break

        # Calculate current FPS.
        actual_fps = frames/(time.time() - last_time)

        # FPS LIMITER...
        # If above desired fps, then put the thread to sleep for a short period of time.
        if DESIRED_FPS < actual_fps:

            sleeptime = (actual_fps - DESIRED_FPS)

            # If sleeptime is too large, shrink it to reduce stuttering.
            if sleeptime > 0.02:
                sleeptime = 0.02
            time.sleep(sleeptime)

if __name__ == "__main__":
    run()
