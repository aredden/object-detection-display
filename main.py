import sys
import time
import cv2
import mss
import numpy as np
import torch
from queue import Queue
from threading import Thread

global DEVICE
global DESIRED_FPS
global DESIRED_SCREEN
global DESIRED_INFERENCE_SIZE
global YOLOV5_MODEL_NAME
"""
GLOBAL VARIABLES THAT CAN BE ADJUSTED!
"""
# Change to 'cpu' for cpu inference.
DEVICE = "cuda"  
# Change to lower number like ~10-30 if you want for slower fps and less cpu / gpu usage.
DESIRED_FPS = 100  
# Screenshot capture area on the screen (single monitor)
DESIRED_SCREEN = {"top": 0, "left": 0, "width": 1920, "height": 1080} 
# Size after resizing from DESIRED_SCREEN
DESIRED_INFERENCE_SIZE = (1280, 720)  
# Model name to pull from torch hub. Models are described / documented on this page: https://github.com/ultralytics/yolov5/releases
YOLOV5_MODEL_NAME = "yolov5m6"

if DEVICE == "cuda":
    try:
        assert torch.cuda.is_available()
    except AssertionError:
        sys.exit("CUDA device isn't available. Either change 'DEVICE' variable to 'cpu'\n"+
        "If you have a CUDA-Capable GPU, I recommend installing pytorch via a command on "+
        "the helper table on this page depending on your OS / package manager / computer platform:\n"+
        "https://pytorch.org/get-started/locally/")

class ScreenShotter(Thread):
    """
    Thread for screenshots with the 'mss' library.
    """

    def __init__(self, screenshot_queue: Queue):
        Thread.__init__(self, daemon=True)
        global DESIRED_SCREEN
        self.images = []
        self.stopped = False
        self.screen = DESIRED_SCREEN
        self.screenshot_queue = screenshot_queue

    def run(self):
        global DESIRED_INFERENCE_SIZE
        with mss.mss() as sct:
            while not self.stopped:
                if self.screenshot_queue.full():
                    time.sleep(0.03)
                    continue
                # Grab image from the screen using the values defined by self.screen
                img_np = np.asarray(sct.grab(self.screen), dtype=np.uint8)
                # Resize the image and put it into the Screenshot queue.
                img_np = cv2.resize(
                    img_np, DESIRED_INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR
                )  # resize
                self.screenshot_queue.put(img_np, block=True, timeout=5)


class ImgInference(Thread):
    """
    Thread for YoloV5 inference and CV2 object outline drawing.
    """
    def __init__(self, screenshot_queue: Queue, display_queue: Queue):
        Thread.__init__(self, daemon=True)
        global YOLOV5_MODEL_NAME
        self.done = False
        self.display_queue = display_queue
        self.screenshot_queue = screenshot_queue
        self.names = []
        self.stopped = False
        self.model = torch.hub.load("ultralytics/yolov5", YOLOV5_MODEL_NAME).half().to(DEVICE)
        self.done = True

    def run(self):
        while not self.stopped:
            # Get next image to infer from the ScreenShotter thread in the background.
            img = self.screenshot_queue.get(
                block=True, timeout=5
            )  # read last screen image

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
                xmin, ymin, xmax, ymax = row[:4]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (90, 10, 255), 1)
                cv2.putText(
                    img,
                    name[:10],
                    (xmin + 4, ymin + 10),
                    cv2.QT_FONT_NORMAL,
                    0.5,
                    (255, 255, 90),
                    thickness=1,
                )

            # Send back the prepared image to display in the main thread.
            self.display_queue.put(img, block=True, timeout=5)

            # Re-Queue the ScreenShotter to get another image from the screen.
            self.screenshot_queue.task_done()



# Main program start thread.
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
        img = display_queue.get(block=True, timeout=5)
        # If this is the first ieration, create the fps counter reference time.
        if frames == 0:
            last_time = time.time()
        frames = frames + 1
        # Display the picture
        cv2.imshow("Fast YoloV5 Object Display", img)

        # Every 15 frames, recalculate the fps.
        if frames % 15 == 0:
            # Cleaar screen
            sys.stdout.write("\033[2J\033[1;1H")
            sys.stdout.flush()

            # Measure FPS, show queue sizes for debugging purposes
            sys.stdout.write(
                f"fps: {frames/(time.time() - last_time)}, dqsize: {display_queue.qsize()}, qsize:{screenshot_queue.qsize()}\n"
            )
            sys.stdout.flush()

        # Tell inference thread to get another frame.
        display_queue.task_done()

        # Press "q" to quit.
        if cv2.waitKey(4) & 0xFF == ord("q"):
            # Destroy all opencv windows when 'Q' is pressed.
            cv2.destroyAllWindows()
            screenShotter.stopped = True
            inference.stopped = True
            break

        # Calculate current FPS.
        actual_fps = frames / (time.time() - last_time)

        # FPS LIMITER...
        # If above desired fps, then put the thread to sleep for a short period of time.
        if DESIRED_FPS < actual_fps:

            sleeptime = actual_fps - DESIRED_FPS

            # If sleeptime is too large, shrink it to reduce stuttering.
            if sleeptime > 0.02:
                sleeptime = 0.02
            time.sleep(sleeptime)


if __name__ == "__main__":
    run()