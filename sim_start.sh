#!/bin/bash

#IMAGE=docker-images.tra.ai/tra-ue-core
IMAGE=sim-image
CONTAINER_NAME=sim

DOCKER_VISUAL_NVIDIA="
-e QT_X11_NO_MITSHM=1
-e DISPLAY
-v /tmp/.X11-unix:/tmp/.X11-unix "

xhost +local:root

docker run --privileged --runtime=nvidia -it --rm $DOCKER_VISUAL_NVIDIA \
--add-host=apt.tra.ai:172.31.1.61 \
-p 7000-7999:7000-7999 \
-v /dev:/dev \
--name $CONTAINER_NAME \
$IMAGE /bin/bash



# apt-get update --allow-insecure-repositories && apt-get install -y tra-simulator --allow-unauthenticated
# ldconfig
# su -c "trasimulator" -m "ue4"


