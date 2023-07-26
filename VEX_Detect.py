import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt
import threading
import Jetson.GPIO as GPIO


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

triballArray = []


def greenDetect(num):
	global triballArray
	try:
	    while True:

		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
		    continue

		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

		#Green Detection
		hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
		# Defining lower and upper bound HSV values
		lower = np.array([50, 100, 100])
		upper = np.array([70, 255, 255])
	  
		# Defining mask for detecting color
		mask = cv2.inRange(hsv, lower, upper)

		#define kernel size  
		kernel = np.ones((7,7),np.uint8)

		# Remove unnecessary noise from mask

		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

		# Segment only the detected region
		segmented_img = cv2.bitwise_and(color_image, color_image, mask=mask)

		# Find contours from the mask

		contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

		triballArray = []

		for c in contours:
		    # Obtain bounding rectangle to get measurements
		    x,y,w,h = cv2.boundingRect(c)

		    # Find centroid
		    M = cv2.moments(c)
		    cX = int(M["m10"] / M["m00"])
		    cY = int(M["m01"] / M["m00"])

		    # Draw the contour and center of the shape on the image
		    cv2.rectangle(color_image, (x,y),(x+w,y+h),(36,255,12), 4)
		    cv2.circle(color_image, (cX, cY), 5, (320, 159, 22), -1)

		    #Get the depth and write to the screen
		    depth = depth_frame.get_distance(cX, cY)
		    s = "Depth: " + str(depth)
		    color_image = cv2.putText(color_image, s, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

	            triballArray.append([x, y, depth])

		# Showing the output
		if len(triballArray) > 0:
		    print(triballArray)

		cv2.imshow("Output", output) # Colored Mask
	      
		#Create the viewing stack
		images = np.hstack((color_image, depth_colormap))

		# Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', images) #Camera and Depth Map
		cv2.imshow('Mask', mask) #Black and White Mask
		cv2.waitKey(1)

	finally:

	    # Stop streaming
	    pipeline.stop()

def sendGreenData(channel):
	GPIO.setmode(GPIO.BOARD)
	GPIO.setwarnings(False)
	GPIO.setup(channel, GPIO.OUT)

	
	


greenDetectThread = threading.Thread(target=greenDetect, args=(1,))

greenDetectThread.start()

greenDetectThread.join()
