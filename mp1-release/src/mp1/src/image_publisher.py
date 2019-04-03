#!/usr/bin/env python
import tf
import rospy
from std_msgs.msg import Float64MultiArray,Float32MultiArray,Float32
from sensor_msgs.msg import Image
import numpy as np
import pdb
import signal
#import roslib
#roslib.load_manifest('my_package')
import sys
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from ImageCallback import Image_Callback

class image_converter:
	def __init__(self):
		self.image_sub = rospy.Subscriber("/vehicle_ZDQ4cW5k/camera/image_raw",Image,self.callback, queue_size = 1)
		self.image_pub = rospy.Publisher("/lane_detection", Image, queue_size=1)
		self.bridge = CvBridge()
	def callback(self,data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
		except CvBridgeError, e:
			print e
		cv_image = Image_Callback(cv_image)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "rgb8"))
		except CvBridgeError, e:
			print e




def main(args):
	ic = image_converter()
	rospy.init_node('image_converter', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "Shutting down"
	cv.DestroyAllWindows()

if __name__ == '__main__':

	main(sys.argv)
