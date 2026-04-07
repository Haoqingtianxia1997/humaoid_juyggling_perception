# build docker container
```bash
cd ~/zed_ws/src/zed-ros2-wrapper/docker
chmod +x ./desktop_build_dockerfile_from_sdk_ubuntu_and_cuda_version.sh
./desktop_build_dockerfile_from_sdk_ubuntu_and_cuda_version.sh ubuntu-22.04 cuda-12.6.3 zedsdk-5.0.7

```

# open docker
```bash
docker run -it --rm   --gpus all   --net=host   --privileged   -v /dev:/dev   -v /tmp/.X11-unix:/tmp/.X11-unix   -e DISPLAY=$DISPLAY   zed_ros2_desktop_u22.04_sdk_5.0.7_cuda_12.6.3

```
# open zed wrapper in docker
```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
apt update
apt install -y ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2 publish_tf:=false publish_map_tf:=false
```

# launch optitrack
```bash
cd ~/mocap_ws
source install/setup.bash 

ros2 launch natnet_ros2 natnet_ros2.launch.py   serverIP:=192.168.2.11   clientIP:=192.168.2.17   serverType:=multicast   pub_rigid_body:=true   activate:=true

ros2 topic echo /juggling_ball/pose

```