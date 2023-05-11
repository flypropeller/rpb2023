#!/usr/bin/env python3
# fly propeller 1등 가자
import cv2
import rospy
import numpy as np

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

class DetermineColor:
    def __init__(self):
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.color_pub = rospy.Publisher('/rotate_cmd', Header, queue_size=10)
        self.bridge = CvBridge()
        self.flag = 0
        self.mask = 0
        self.approx = 0

    def callback(self, data):
        try:
            # listen image topic
            img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            cv2.imshow('Image', img)
            cv2.waitKey(1)
            # prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP

            # Determine background color

            # Detect the screen at first
            if self.flag == 0:

                # Grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Apply edge detection
                edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

                # Find contours
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the contour with the largest area (i.e. the screen)
                largest_contour = max(contours, key=cv2.contourArea)

                # Approximate the contour with a polygon
                epsilon = 0.1 * cv2.arcLength(largest_contour, True)
                self.approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Extract the screen region
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [self.approx], 0, (255, 255, 255), -1)

            # Extract screen from image
            screen = cv2.bitwise_and(img, img, mask=mask)

            # HSV color
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)

            # Define the red mask
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            lower_red = np.array([160, 100, 100])
            upper_red = np.array([189, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

            mask_red = cv2.bitwise_or(mask1, mask2)

            # Define the blue mask
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            # Count red, blue pixels
            num_red_pixels = cv2.countNonZero(mask_red)
            num_blue_pixels = cv2.countNonZero(mask_blue)

            # Define the mask that finds all pixels
            lower_none = np.array([1, 1, 1])
            upper_none = np.array([255, 255, 255])
            mask_none = cv2.inRange(hsv, lower_none, upper_none)

            # Count all pixels
            num_none_pixels = cv2.countNonZero(mask_none)

            # Determine the rotating direction
            if num_red_pixels > num_none_pixels / 2:
                msg.frame_id = '-1'
            elif num_blue_pixels > num_none_pixels / 2:
                msg.frame_id = '+1'
            else:
                msg.frame_id = '0'

            # Draw contours
            cv2.polylines(img, [self.approx], True, (0, 255, 0), thickness=2)

            # Publish color_state
            self.color_pub.publish(msg)
            self.flag = 1

        except CvBridgeError as e:
            print(e)

    def rospy_shutdown(self, signal, frame):
        rospy.signal_shutdown("shut down")
        sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('CompressedImages1', anonymous=False)
    detector = DetermineColor()
    rospy.spin()