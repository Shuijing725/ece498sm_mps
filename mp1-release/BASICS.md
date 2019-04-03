# MP0 Files
These are programming files for the Machine Problem-0. The essential idea of this MP is to get acquainted with Roobotics Operating System (ROS) and it's interfacing with RightHook. 

# ROS

Robotics Operating System or ROS is short was project that came out of Stanford in the early 2007. The idea was to make a simple easy to use software stack that makes it easier for different modules in robots like perception, control, planning to work in-sync with each other. 

There are a couple of interesting terminologies associated with ROS which we describe as follows, 
* **Nodes** Any Piece of executable code in Python is called Node. Consider this simple example, wherein we are trying to stop our car at a four-way intersection as soon as we detect a stop sign
* **Topics** In order to detect these stop signs, our piece of code (ROS Node) essentially needs to access to sensor data that is mounted on the car. In ROS terminology, these sensor data is available on a virtual cloud called as ROS topics
* **ROS Messages** In ROS, nodes never directly communicate with each other. Instead they communicate anonymously by streaming and subscribing data that is being published on ROS topics. ROS messages is just the data structure associated with ros topics. 


# ROS Semantics
Consider for example [vehicle_control_example.py](src/vehicle_control_example.py) as a part of your MP0 code. In this particular MP, the vehicle_control node takes data from the camera mounted on the car(*vehicle_id\image_raw*) does some processing and comes up with control commands for the car (*throttle, brake and steering*) and these values are published on *vehicle_id/actuation/cmd* which in turn is used by the right-hook software to run the simulation.
![Example of ROS architecture](https://i.imgur.com/RKyHN0Q.png)

To understand this piece of code. let us go by this line by line. 

* line 1
```python
#!/usr/bin/env python
```
This line essentially ensures that this file be executed as a python script

* line 4
```python
import rospy
```
This import helps use python semantics for ros-python wrapper

* line 5
```python
import sensor_msgs import Image
```
This import helps us use the Image message type used by the *vehicle_id\image_raw* topic

* line 7
```python
from cv_bridge import CvBridge, CvBridgeError
```
This import helps convert Image messages to opencv compatible data structure

* line 10-13
```python
# RightHook simulation control service defs
from rh_msgs.srv import SimControl, SimControlRequest

# RightHook actuation message
from rh_msgs.msg import VehicleActuation
```
These are the service-control requests that are required to run right-hook simulation. We shouldn't worry about it now. 

* line 47 
```python
rospy.init_node('vehicle_controller', anonymous=True)
```
This line of code essentially initialises this piece of code as a ros node with the name, *vehicle_controller*. Note that, *anonymous=True* option ensures this node name is unique and does not face any conflict issues

* line 53
```python
actuation_publisher = rospy.Publisher(
		'/vehicle_ZDQ4cW5k/actuation/cmd',
		VehicleActuation,
		queue_size=0)
```
This line essentially initialises the topic, */vehicle_ZDQ4cW5k/actuation/cmd* as a publisher. The data structure of the message expected by this publisher is given by **VehicleActuation**. **queue_size=0** This is the size of the outgoing message queue used for publishing.

* line 58
```python 
stop_sign_detector = rospy.Subscriber("/vehicle_ZDQ4cW5k/camera/image_raw",Image,stop_sign.callback, queue_size = 1)
```
Line essentially subsrcibes to the data recieved on */vehicle_ZDQ4cW5k/camera/image_raw* topic. the next argument in this function tells us the message type of this topic (Image message) and the third argument associated is the call-back function, which does the processing of this image as soon as data is available in this topic. In this particular example, *stop_sign.callback* is the function call and it tells us whether the stop sign has been detected

* line 102
```python
actuation_publisher.publish(actuation_msg)
```
After the processing performed by the code. This line actually published data to the publisher topic which is used by right-hook backend to run a simulation step. 

## References
1. [http://wiki.ros.org/ROS/Tutorials/UnderstandingNodes](http://wiki.ros.org/ROS/Tutorials/UnderstandingNodes)
2. [http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29)
