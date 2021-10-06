# Real-Time & Fast Object Detection and Display

Uses two threads and two queue's for fast model inference without spending time waiting for screenshots or images to be rendered to an opencv window.

## How it works

### Screenshotter Thread: 
+ Do screenshot -> 
+ Resize screenshot ->  
+ Add to requires-inference queue -> 
+ Repeat ->
### Inference Thread: 
+ Get image from requires-inference queue -> 
+ Feed image to model -> 
+ Process result boxes and labels and draw them on image -> 
+ Add to prepared-screenshot queue -> 
+ Repeat ->
### Main Thread: 
+ Wait for prepared screenshots from prepared-screenshot queue -> 
+ Show screenshot -> 
+ Repeat ->

## Honestly idk what to call this section

My Laptop with an 80 watt 3060 mobile was able to achieve 38 fps with the following global variable settings, using python 3.9.6.
```py
# Change to 'cpu' for cpu inference.
DEVICE = "cuda"  
# Change to lower number like ~10-30 if you want for slower fps and less cpu / gpu usage.
DESIRED_FPS = 100  
# Screenshot capture area on the screen (single monitor)
DESIRED_SCREEN = {"top": 0, "left": 0, "width": 1920, "height": 1080} 
# Size after resizing from DESIRED_SCREEN
DESIRED_INFERENCE_SIZE = (1280, 720)  
# Model name to pull from torch hub. Models are described / documented on this page: https://github.com/ultralytics/yolov5
YOLOV5_MODEL_NAME = "yolov5m6"
```

## References

Fast Object Detection: [yolov5](https://github.com/ultralytics/yolov5)
