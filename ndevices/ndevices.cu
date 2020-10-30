#include <cuda.h>
#include <stdio.h> 

int main() {
  int driver_version = 0, runtime_version = 0;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);

  printf("Driver Version: %d\n Runtime Version: %d\n", \
    driver_version, runtime_version);

  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  MultiProccessor: %d\n", prop.multiProcessorCount);
    printf("  Total GlobalMemory mb: %zu\n", prop.totalGlobalMem/(1024*1024));
    printf("  Total ConstMemory mb: %zu\n", prop.totalConstMem/(1024*1024));
    printf("  ---\n");

    printf("  sharedMemPerBlock: %zu\n", prop.sharedMemPerBlock);
    printf("  regsPerBlock: %d\n", prop.regsPerBlock);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  ---\n");

    printf("  sharedMemPerMultiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("  regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  ---\n");

    printf("  warpSize: %d\n", prop.warpSize);
    printf("  memPitch: %d\n", prop.memPitch);
    printf("  maxThreadsDim[3]: %d\n", prop.maxThreadsDim[3]);
    printf("  maxGridSize[3]: %d\n", prop.maxGridSize[3]);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
