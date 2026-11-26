FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV LIBGL_ALWAYS_SOFTWARE=1

# Install dependencies (manual §2.4, step 3)
# Ubuntu archive mirrors occasionally return transient 5xx responses during a
# long dependency install.  Let apt retry those downloads before failing a
# complete image build.
RUN apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y \
    libjsoncpp-dev \
    ros-noetic-ddynamic-reconfigure \
    libpcap-dev \
    ros-noetic-serial \
    ros-noetic-ros-numpy \
    python3-pcl \
    libqt5serialport5-dev \
    build-essential \
    zlib1g-dev \
    libx11-dev \
    libusb-1.0-0-dev \
    freeglut3-dev \
    liblapacke-dev \
    libopenblas-dev \
    libatlas-base-dev \
    cmake \
    git \
    libssl-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    g++ \
    python3-pip \
    python3-opencv \
    libopenvr-dev \
    unzip \
    ros-noetic-cv-bridge \
    ros-noetic-rviz \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Keep Ubuntu's OpenCV, which must stay ABI-compatible with ROS Noetic's
# cv_bridge.  Installing PyPI's current opencv-python here overwrites it and
# makes fisheye image conversion fail at runtime.
RUN pip3 install h5py scipy open3d

# Clone the ROS source without its full Git history.  A timeout and retries
# avoid a Docker build hanging indefinitely on a stalled GitHub connection.
WORKDIR /root
RUN for attempt in 1 2 3; do \
      git -c http.connectTimeout=30 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 \
        clone --depth 1 https://github.com/jshyunbin/pika_ros.git && break; \
      rm -rf /root/pika_ros; \
      if [ "$attempt" = 3 ]; then exit 1; fi; \
      sleep 10; \
    done

# Build librealsense 2.55.1 from source bundled in pika_ros/source/ (manual §2.4, step 5)
WORKDIR /root/pika_ros/source
RUN unzip librealsense-2.55.1.zip && tar -xzf curl-7.75.0.tar.gz


WORKDIR /root/pika_ros/source/librealsense
RUN sed -i \
    's|URL "/home/agilex/curl-7.75.0"|URL "/root/pika_ros/source/curl-7.75.0"|g' \
    CMake/external_libcurl.cmake
RUN bash install.bash

# The pre-built realsense2_camera nodelet in install.zip was compiled against librealsense 2.50.
# Create a symlink so it can find the 2.55.1 library built above.
RUN ln -sf /usr/local/lib/librealsense2.so.2.55 /usr/local/lib/librealsense2.so.2.50 && ldconfig

# Extract the pre-built pika_ros install tree (manual §2.4, step 6)
WORKDIR /root/pika_ros/source
RUN unzip install.zip -d /root/pika_ros/ && chmod 777 -R /root/pika_ros/install/ && \
    SCRIPTS=/root/pika_ros/install/share/sensor_tools/scripts && \
    sed -i \
        '/udevadm\|sensor_serial\.rules\|sensor_fisheye\.rules\|gripper_serial\.rules\|gripper_fisheye\.rules/d' \
        $SCRIPTS/start_single_sensor.bash $SCRIPTS/start_single_gripper.bash && \
    sed -i \
        's|cd \$SCRIPT_DIR/\.\./install/share/sensor_tools/scripts|cd $SCRIPT_DIR|g; s|source \$SCRIPT_DIR/\.\./install/setup\.bash|source /root/pika_ros/install/setup.bash|g' \
        $SCRIPTS/start_single_sensor.bash $SCRIPTS/start_single_gripper.bash \
        $SCRIPTS/start_multi_sensor.bash $SCRIPTS/start_multi_gripper.bash \
        $SCRIPTS/start_sensor_gripper.bash

# Create libsurvive config directory so calibration persists within the container
RUN mkdir -p /root/.config/libsurvive

# Source ROS environments on login (manual §2.4, step 7)
RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc && \
    echo 'source /root/pika_ros/install/setup.bash' >> /root/.bashrc

# Entrypoint ensures ROS env is active for both interactive shells and exec invocations
RUN printf '#!/bin/bash\nsource /opt/ros/noetic/setup.bash\nsource /root/pika_ros/install/setup.bash\nexec "$@"\n' \
    > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
