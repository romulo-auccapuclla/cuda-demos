#include <cuda.h>
#include <stdio.h> 
#include <iostream>
#include <string>

using namespace std;

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
    int cudaCores = 0;
    int SM = prop.multiProcessorCount;
    int major = prop.major;
    int minor = prop.minor;
    string arch = "";
    switch (major) {
    case 1:
        arch = "TESLA";
        cudaCores = 8;
        break;
    case 2:
        arch = "FERMI";
        if (minor == 0)
            cudaCores = 32;
        else
            cudaCores = 48;
        break;
    case 3:
        arch = "KEPLER";
        cudaCores = 192;
        break;
    case 5:
        arch = "MAXWELL";
        cudaCores = 128;
        break;
    case 6:
        arch = "PASCAL";
        if ((minor == 1) || (minor == 2)) cudaCores = 128;
        else if (minor == 0) cudaCores = SM * 64;
        else printf("Unknown device type\n");
        break;
    case 7:
        if ((minor == 0) || (minor == 2)) {
            arch = "VOLTA";
            cudaCores = 384;
            //tensorCores = 48;
        }
        if (minor == 5) arch = "TURING";
        if ((minor == 0) || (minor == 5)) cudaCores = 64;
        else printf("Unknown device type\n");
        break;
    case 8:
        arch = "AMPERE";
        if (minor == 0) cudaCores = 64;
        else printf("Unknown device type\n");
        break;
    default:
        //ARQUITECTURA DESCONOCIDA
        cudaCores = 0;
        printf("!!!!!dispositivo desconocido!!!!!\n");
    }

    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    cout << "  Architecture: " << arch << endl;
    printf("  Compute Capability: %d.%d\n", major, minor);
    printf("  MultiProccessors: %d\n", SM);
    printf("  CUDA Cores (%dx%d): %d\n", cudaCores, SM, cudaCores*SM);
    printf("  GlobalMemory (total): %zu MiB\n", prop.totalGlobalMem/(1024*1024));
    printf("  ConstMemory (total): %zu KiB\n", prop.totalConstMem/1024);

    printf("  sharedMemPerMultiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("  regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);

    printf("  sharedMemPerBlock: %zu\n", prop.sharedMemPerBlock);
    printf("  regsPerBlock: %d\n", prop.regsPerBlock);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("    x = %d\n", prop.maxThreadsDim[0]);
    printf("    y = %d\n", prop.maxThreadsDim[1]);
    printf("    z = %d\n", prop.maxThreadsDim[2]);
    printf("  maxThreadsDim: %d\n", prop.maxThreadsDim[3]);
    printf("  maxGridSize: %d\n", prop.maxGridSize[3]);
    printf("    x = %d\n", prop.maxGridSize[0]);
    printf("    y = %d\n", prop.maxGridSize[1]);
    printf("    z = %d\n", prop.maxGridSize[2]);

    printf("  warpSize: %d\n", prop.warpSize);
    printf("  memPitch: %zu\n", prop.memPitch);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
