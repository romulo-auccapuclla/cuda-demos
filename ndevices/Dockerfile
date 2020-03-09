ARG CUDA_VER=9.2
ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu${UBUNTU_VERSION} as builder
WORKDIR /app
COPY ./ndevices.cu /app/
RUN nvcc /app/ndevices.cu -o ndevices && /app/ndevices

FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu${UBUNTU_VERSION}
COPY --from=builder ./app/ndevices /
COPY ./ndevices.sh /
CMD ["/ndevices.sh"]
