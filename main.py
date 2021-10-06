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
DESIRED_FPS = 40
# Screenshot capture area on the screen (single monitor)
DESIRED_SCREEN = {"top": 0, "left": 0, "width": 1920, "height": 1080} 
# Size after resizing from DESIRED_SCREEN
DESIRED_INFERENCE_SIZE = (960, 540)  
# Model name to pull from torch hub. Models are described / documented on this page: https://github.com/ultralytics/yolov5/releases
YOLOV5_MODEL_NAME = "yolov5m6"

if DEVICE == "cuda":
    try:
        assert torch.cuda.is_available()
    except AssertionError:
        sys.exit("CUDA device isn't available. Either change 'DEVICE' variable to 'cpu', or..\n"+
        "If you have a CUDA-Capable GPU, I recommend installing pytorch via a command on "+
        "the helper table on this page depending on your OS / package manager / computer platform:\n"+
        "https://pytorch.org/get-started/locally/")

class ScreenShotter(Thread):
    """
    Thread for screenshots with the 'mss' library.
    """

    def __init__(self, screenshot_queue: Queue):
        Thread.__init__(self, daemon=True)
        self.stopped = False
        self.screenshot_queue = screenshot_queue

    def run(self):
        global DESIRED_INFERENCE_SIZE
        global DESIRED_SCREEN
        # Initialize the mss screenshot tool
        with mss.mss() as sct:
            # Begin thread loop
            while not self.stopped:
                # If screenshot queue is full, wait for free space to add new image.
                if self.screenshot_queue.full():
                    time.sleep(0.03)
                    continue
                # Grab image from the screen using the values defined by self.screen
                img_np = np.asarray(sct.grab(DESIRED_SCREEN), dtype=np.uint8)
                # Resize the image and put it into the Screenshot queue.
                img_np = cv2.resize(
                    img_np, DESIRED_INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR
                )  # resize
                self.screenshot_queue.put(img_np, block=True, timeout=5)


class ImgInference(Thread):
    """
    Thread for YoloV5 inference and CV2 object outline drawing.
    """
    def __init__(self, screenshot_queue: Queue, draw_queue: Queue):
        Thread.__init__(self, daemon=True)
        global YOLOV5_MODEL_NAME
        global DEVICE
        self.done = False
        self.draw_queue = draw_queue
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
                # Text labels array
                self.names = dat.names
                # Take the list of [minx, miny, maxx, maxy, confidence, text-label-index]
                # result values "xyxy", and parse result as integer, send from gpu to cpu,
                # and generate as a numpy array.
                dat = dat.xyxy[0].int().cpu().numpy()
            else:
                # If label names exist, make prediction slightly more efficiently.
                
                # Generate the detections and predictions on the gpu as tensors, and
                # send to cpu -> numpy array in one line.
                dat = self.model(img).xyxy[0].int().cpu().numpy()
            
            self.draw_queue.put({'img': img, 'dat': dat, 'names': self.names})
            self.screenshot_queue.task_done()
            # Re-Queue the ScreenShotter to get another image from the screen.
            

class DrawObjects(Thread):
    """
    Thread for drawing object detection boxes and labels onto screenshots.
    """
    def __init__(self, draw_queue:Queue, display_queue: Queue):
        Thread.__init__(self, daemon=True)
        self.stopped = False
        self.draw_queue = draw_queue
        self.display_queue = display_queue

    def run(self):

        while not self.stopped:
            outputs = self.draw_queue.get(block=True, timeout=5)
            img = outputs['img']
            dat = outputs['dat']
            names = outputs['names']
            # Draw predicted bounding boxes and labels onto image.
            for row in dat:
                # The name of the object in this 'box' is the last value in the dat array, 
                # which is in the form of the integer index of the text label in the 'names' list.
                name = names[row[-1]]
                # First four values in the row array represent the left-top, and right-bottom points respectively.
                xmin, ymin, xmax, ymax = row[:4]
                # Draw rectangle onto image, and the label within it.
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (90, 10, 255), 1)
                img = cv2.putText(
                    img, # Image
                    name[:10], # Label
                    (xmin + 4, ymin + 10), # Where to draw text
                    cv2.QT_FONT_NORMAL, # Font type
                    0.5, # Scale
                    (255, 255, 90), # Color
                    thickness=1, # Thickness of font
                )
            # Send back the prepared image to display in the main thread.
            self.display_queue.put(img, block=True, timeout=5)
            self.draw_queue.task_done()


# Main program start thread.
def run():

    # Queue for storing screenshots to send to inference thread.
    screenshot_queue = Queue(maxsize=3)
    # Queue for drawing labels onto screenshots.
    draw_queue = Queue(maxsize=3)
    # Queue for sending labeled screenshots to be displayed in main thread.
    display_queue = Queue(maxsize=3)
    # Thread for retrieving screenshots from the screen.
    screenShotter = ScreenShotter(screenshot_queue)
    # Thread for drawing labels onto screenshots.
    draw = DrawObjects(draw_queue, display_queue)
    # Thread for object detection and labeling.
    inference = ImgInference(screenshot_queue, draw_queue)
    # Wait for pytorch model to be loaded.
    while not inference.done:
        time.sleep(0.05)

    # Initialize fps counter.
    frames = 0
    
    # Launch threads.
    inference.start()
    draw.start()
    screenShotter.start()
    # Begin the main loop infinetly until 'Q' is pressed.
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
                f"FPS: {frames/(time.time() - last_time)},\n"+
                f"Display Queue Size: {display_queue.qsize()},\n"+
                f"Screenshot Queue Size: {screenshot_queue.qsize()},\n"+
                f"DrawObjects Queue Size: {draw_queue.qsize()}\n"
            )
            sys.stdout.flush()

        # Tell inference thread to get another frame.
        display_queue.task_done()

        # Press "q" to quit.
        if cv2.waitKey(4) & 0xFF == ord("q"):
            # Destroy all opencv windows when 'Q' is pressed.
            cv2.destroyAllWindows()
            # Stop screenshot / inference threads.
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
