#!/bin/bash

CUDA_VER=9.2
if [ ! -z $1 ]; then
  CUDA_VER=$1
fi

docker build -t mario21ic/ndevices:cuda${CUDA_VER}-v1 --build-arg CUDA_VER=${CUDA_VER} ./

docker run --rm --gpus=all mario21ic/ndevices:cuda${CUDA_VER}-v1
