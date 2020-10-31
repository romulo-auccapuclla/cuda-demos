// Hilos y Bloques
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK 10
#define N 25

// Kernel Multibloque
__global__
void suma(int *vector_1, int *vector_2, int*vector_suma, int n)
{
    // identificador de hilo
    int myID = threadIdx.x + blockDim.x * blockIdx.x;

    // solo para n hilos
    if (myID < n) {
        // inicializamos el vector 2
        vector_2[myID] = (n -1) - myID;
        // escritura de resultados
        vector_suma[myID] = vector_1[myID] + vector_2[myID];
    }
}

int main(int argc, char** argv) {
    // declaraciones
    int *hst_vector1, *hst_vector2, *hst_resultado;
    int *dev_vector1, *dev_vector2, *dev_resultado;

    // reserva en el host
    hst_vector1 = (int*)malloc(N * sizeof(int));
    hst_vector2 = (int*)malloc(N * sizeof(int));
    hst_resultado = (int*)malloc(N * sizeof(int));

    // reserva en el device
    cudaMalloc((void**)&dev_vector1, N * sizeof(int));
    cudaMalloc((void**)&dev_vector2, N * sizeof(int));
    cudaMalloc((void**)&dev_resultado, N * sizeof(int));

    // inicializacion de vectores
    for (int i=0; i<N; i++) {
        hst_vector1[i] = i;
        hst_vector2[i] = 0;
    }

    // copia de datos hacia el device
    cudaMemcpy(dev_vector1, hst_vector1, N*sizeof(int), cudaMemcpyHostToDevice);

    // calculamos nro de bloques
    int bloques = N / BLOCK;
    /* Si el tamano del vector no es multiplo del 
    tamano del bloque, lanzamos otro bloque
    (tendremos hilos sobrantes) */
    if ((N % BLOCK) != 0) {
        bloques = bloques + 1;
    }

    printf("> Vector de %d elementos\n", N);
    printf("> Lanzamiento con %d bloques de %d hilos (%d hilos)\n", bloques, BLOCK, bloques*BLOCK);

    // Lanzamiento del Kernel
    suma <<< bloques, BLOCK >>> (dev_vector1, dev_vector2, dev_resultado, N);

    // recogida de datos desde el device
    cudaMemcpy(hst_vector2, dev_vector2, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hst_resultado, dev_resultado, N*sizeof(int), cudaMemcpyDeviceToHost);

    // impresion de resultados
    printf("VECTOR 1:\n");
    for (int i=0; i<N; i++) {
        printf("%.2d ", hst_vector1[i]);
    }
    printf("\n");

    printf("VECTOR 2:\n");
    for (int i=0; i<N; i++) {
        printf("%.2d ", hst_vector2[i]);
    }
    printf("\n");

    printf("SUMA:\n");
    for (int i=0; i<N; i++) {
        printf("%.2d ", hst_resultado[i]);
    }
    printf("\n");

    printf("****");
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 0;
}
