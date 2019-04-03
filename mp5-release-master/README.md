#  Introduction

In this MP you are going to implement a RRT path planner for Autonomous cars. In the first part you will first learn to implement a simple controller for maneuvering the car from point A to point B. In the second part, you are going to use this controller for writing RRT path planning algorithm.

# Installation and Setup

Unlike all other MPs you are **not **going to implement this assignment using  jupyter notebook but you will be editing python scripts.  Clone the assignment repository

<pre>git clone https://gitlab.engr.illinois.edu/GolfCar/mp5-release
cd mp5-release

</pre>

In this particular assignment you are going to specifically edit two files which are listed below

1.  **controller.controller.py (**See Controller synthesis section**)**
2.  **RRT.py (**See RRT section**)**

## Control Synthesis

## Dubin’s model of the car

In the control synthesis part of the Machine Problem you first have to design a controller which takes the car from point A to point B. Before synthesizing the controller, we first need to come up with the control model of the problem.

For newbies here, a control model is the sensitivity of the change of state of the car (for example position) with respect to the control inputs which in this case is the **thrust** ($latex V_R$) and **steering angle** ($latex \delta$). There have been many proposed control models of the car but as far as this MP is concerned we are going to use Dubin's Model for the car which is as follows,

$$\begin{bmatrix}\dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix} =\begin{bmatrix}V_R \cos(\theta) \\ V_R \sin(\theta) \\ \delta \end{bmatrix}$$

[caption id="attachment_670" align="aligncenter" width="584"]![](http://publish.illinois.edu/safe-autonomy/files/2019/03/dubinschematic-1024x712.png) Schematic for Dubin's Car[/caption] In this controller synthesis you are going to implement a proportional controller to drive the car from point **A **to point **B**. To do this you will have to first correct the heading of the car so that it lies in line with point **B** and then you are just supposed to drive straight to point **B**. The algorithm is described as below.

### **Controller Algorithm**

* * *

**Initialise: **$latex \epsilon > 0, K > 0, Dt, V_R$, $latex B = (x_d, y_d)$, $latex init\_state = (x, y, \theta)$ $latex alpha = arctan((y_d - y)/(x_d - x)) - theta$ $latex \textbf{while} (|alpha| > \epsilon):$ $latex \delta = K*(alpha)$ $latex (x, y, \theta) = model(\delta, V_R = 0.0)$ $latex alpha = arctan((y_d - y)/(x_d - x)) - theta$ $latex \textbf{end while}$ $latex Dist = \sqrt{(x_d - x)^2 + (y_d - y)^2}$ $latex Time Steps = \frac{Dist}{V_R*Dt}$ $latex \textbf{For i = 1 to TimeSteps}$ $latex (x, y, \theta) = model(\delta = 0.0, V_R)$ $latex \textbf{end for}$

* * *

You are going to have to implement this controller in the blanked out portion in _controller/controller.py _in the gitlab repository. After implementing this, you will then need to implement the **RRT path planner** as described in the next section.

# RRT

RRT algorithm was proposed by our Prof. Steve LaValle and Dr. James Kuffner in 1998 as an efficient search algorithm in a non-convex space. RRTs were later demonstrated to handle differential constraints so well that roboticist started using this in **Motion Planning**. In this part of the MP you are going to implement a RRT path planning algorithm for a self-driving car problem. RRT algorithm is described in detail below.

# Rapidly-Exploring Random Trees Algorithm

* * *

**Input:** Initial state of the car $latex X_{init} = (x, y, \theta)$, World map $latex G$, $latex X_{goal}$ **Initialise:** $latex q_{new}=\{ \}$, $latex q_{rand} = \{ \}$, $latex G=\{(q_{init},q_{init}) \}$ **Output:** RRT graph **while** $latex q_{new} \neq q_{goal}$ $latex q_{rand} = rand()$ $latex q_{near} = nearest\_neighbour(q_{rand}, G)$ $latex safety = prelim\_verify(q_{rand}, q_{near})$ **while:** $latex safety ==True$ $latex q_{new} = run\_controller(q_{near}, q_{rand}, \Delta q)$ $latex G = G \cup (q_{new}, q_{init})$ **end** **end** **return** $latex path\_from(G, q_{init}, q_{goal})$

* * *

After the full run of the RRT algorithm an example reconstruction would look something like this [![](https://i.imgflip.com/2xfiwl.gif "made at imgflip.com")](https://imgflip.com/gif/2xfiwl)

# **Summary**

In this MP you are first going to implement a lower-level controller for a self-driving car (**controller.controller.py)**. Using this controller you are going to implement the RRT algorithm in **RRT.py** file. After completing the implementation of the RRT algorithm you need to make a concise report answering the following questions

1.  You saw in the file RRT.py that the points in the map is sampled uniformly with a minimum grid size of 10 pixels. Trying running your RRT code on with fineness of 1 pixels and fineness of 100 pixels and observe the results
2.  Instead of the sampling mentioned in the RRT code. Try implementing a goal based sampling. Does the code converge faster in that case? Why/Why not ?
3.  Do you think that RRT always comes up with a 'safe' path? Why or Why not ?

# Grading Evaluation

1.  Controller Implementation (20%)
2.  RRT implementation (60%)
3.  Completing the project report with question responses (20%)
