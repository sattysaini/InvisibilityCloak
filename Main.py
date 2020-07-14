import cv2
import numpy as np
import time


# taking video.mp4 as input.
# Making the output file ready to tak output
capture_video = cv2.VideoCapture("video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.mp4', fourcc, 29.61, (640, 352))

# give the camera to warm up
time.sleep(1)
count = 0
background = 0

# capturing the background in range of 60
# to save the background image
for i in range(60):
    return_val, background = capture_video.read()
    if return_val == False:
        continue

background = np.flip(background, axis=1)  # flipping of the frame

# we are reading from video
while (capture_video.isOpened()):
    return_val, img = capture_video.read()
    if not return_val:
        break
    count = count + 1
    img = np.flip(img, axis=1)

    # convert the image - BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # setting the lower and upper range for mask1
    lower_red = np.array([110, 60, 60])
    upper_red = np.array([110, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # setting the lower and upper range for mask2
    lower_red = np.array([85, 60, 60])
    upper_red = np.array([100, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # combining the masks
    mask1 = mask1 + mask2

    # Refining the mask corresponding to the detected color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3),
                                                            np.uint8), iterations=2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    # Generating the final output and saving it
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    out.write(final_output)
    cv2.imshow("INVISIBILITY CLOAK", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_video.release()
out.release()
cv2.destroyAllWindows()