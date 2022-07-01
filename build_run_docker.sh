#! /bin/sh

nvidia-docker build -t viper-img .

nvidia-docker run --rm --interactive \
  --volume "$PWD:/viper" \
  viper-img \
  ./docker_build_entry.sh

XSOCK=/tmp/.X11-unix

nvidia-docker run --rm --interactive \
  --volume "$PWD:/viper" \
  --workdir /viper/build \
  -e DISPLAY=$DISPLAY \
  -v $XSOCK:$XSOCK \
  -v $HOME/.Xauthority:/root/.Xauthority \
  viper-img \
  ./demo
