#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 16

int main(int argc, char** argv) {
    // declaraciones
    float *hst_A, *hst_B;
    float *dev_A, *dev_B;
    
    // reserva en el host
    hst_A = (float*)malloc(N * sizeof(float));
    hst_B = (float*)malloc(N * sizeof(float));

    // reserva en el device
    cudaMalloc((void**)&dev_A, N * sizeof(float));
    cudaMalloc((void**)&dev_B, N * sizeof(float));

    // inicializacion
    for (int i=0; i<N; i++) {
        hst_A[i] = (float)rand() / RAND_MAX;
        hst_B[i] = 0;
    }

    // copia de datos
    cudaMemcpy(dev_A, hst_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, dev_A, N*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(hst_B, dev_B, N*sizeof(float), cudaMemcpyDeviceToHost);

    // muestra de resultados
    printf("ENTRADA (hst_A):\n");
    for (int i=0; i<N; i++) {
        printf("%.2f ", hst_A[i]);
    }
    printf("\n");

    printf("SALIDA (hst_B):\n");
    for (int i=0; i<N; i++) {
        printf("%.2f ", hst_B[i]);
    }
    printf("\n");

    // liberacion de recursos
    cudaFree(dev_A);
    cudaFree(dev_B);

    return 0;
}
