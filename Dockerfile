from nvidia/cudagl:10.2-base-ubuntu18.04

env NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

workdir /viper

run apt-get update --yes
run apt-get install --yes \
    cuda-toolkit-10-2 \
    g++ \
    cmake \
    xorg-dev \
    libboost-all-dev \
    libglew-dev \
    libcgal-dev \
    libtbb-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libgl1-mesa-dri
