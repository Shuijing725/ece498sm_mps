source /opt/ros/kinetic/setup.bash
CONTAINER_ID=$(docker ps -aq -f name=$USER)
export HOST_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}' $CONTAINER_ID)
export MY_IP=$HOST_IP
export CONTAINER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' $CONTAINER_ID)
export ROS_MASTER_URI=http://$CONTAINER_IP:11311
export ROS_IP=$HOST_IP
export ROS_HOSTNAME=$HOST_IP
