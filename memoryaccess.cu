#include <iostream>
#include <math.h>

__constant__ int constant_values[100];

__global__ void test_kernel(int* d_array)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<100; i++) {
        d_array[idx] = d_array[idx] + constant_values[i];
    }
    return;
}


int main(int argc, char** argv)
{
    std::cout << "Starting" << std::endl;
    int size = 100*sizeof(int);
    int* d_array;
    int h_angle[360];
    int BLOCK_SIZE = 64;
    cudaError_t cudaStatus;

    //std::srand(std::time(0));
    // Reserva espacio device memory
    cudaMalloc((void**)&d_array, sizeof(int)*size);
    // Inicializacion memoria device a 0
    cudaMemset(d_array, 0, sizeof(int)*size);
    // Inicializacion en el host la informacion constante
    for (int i=0; i<100; i++) {
        h_angle[i] = std::rand();
        // Copia datos a memoria constante en CUDA
        cudaMemcpyToSymbol(constant_values, h_angle, sizeof(int)*100);
        test_kernel<<<100/BLOCK_SIZE,BLOCK_SIZE>>>(d_array);
        // Comprueba errores llamada al kernel ( se han obviado el resto de comprobaciones)
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));
            return 1;
        }
    }
    // liberamos memoria device
    cudaFree(d_array);
    std::cout << "Finishing" << std::endl;
    return 0;
}
