#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def setup():
	rospy.init_node('camera_interpreter')

	rospy.Subscriber('/R1/pi_camera/image_raw', Image, linefind)
	rospy.spin()

#Callback function receives image and proccesses it into a command
def linefind(data):
	bridge = CvBridge()
	cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

	#this is where the code from lab 2 begins#
	imgray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
	imgray_cropped = imgray[-100:-1] #y-values first

	ret,thresh = cv2.threshold(imgray_cropped,127,255,0)
	thresh_inverted = cv2.bitwise_not(thresh)

	#having issues doing decision making for when line is not detected
	contours,hierarchy = cv2.findContours(thresh_inverted, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	try:
		biggestContour=max(contours, key=cv2.contourArea)
	except:
		forward = 0
		turn = 3
		sendDriveCommand(forward, turn)
	if cv2.contourArea(biggestContour) < 1000:
		forward = 0
		turn = 3
		sendDriveCommand(forward, turn)
		return

	moment_array=cv2.moments(biggestContour)
	try:
		cx = int(moment_array['m10']/moment_array['m00'])
	except ZeroDivisionError:
		#how do i do nothing
		print("divide by zero in moments")
	width = len(imgray[0])
	rospy.loginfo("Width: " + str(width))

	#perhaps try implementing primitive PID
	if cx <= width / 2:
		forward = 0.5
		turn = 1
		sendDriveCommand(forward, turn)
	elif cx > width / 2:
		forward = 0.5
		turn = -1
		sendDriveCommand(forward, turn)
	rospy.loginfo("Forward value: " + str(forward)  + "Turn value: " + str(turn))

#called from lineDrive, publishes to cmd_vel so the car can drive.
def sendDriveCommand(forwardAmount, turnAmount):
	pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
	move = Twist()
	move.linear.x = forwardAmount
	move.angular.z = turnAmount
	pub.publish(move)

if __name__ == '__main__':
	setup()

#while not rospy.is_shutdown():
# pub.publish(move)
#	interpret()
#	rate = rospy.Rate(2)
#	rate.sleep()