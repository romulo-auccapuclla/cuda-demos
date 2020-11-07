// Temporizador de un Kernel
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

# define N 24

// Funcion que suma de dos vectores de n elementos
__global__
void suma(int *vector_1, int *vector_2, int*vector_suma, int n)
{
    // identificador de hilo
    int myID = threadIdx.x;

    // obtenemos el vector invertido
    vector_2[myID] = vector_1[n -1 - myID];

    // escritura de resultados
    vector_suma[myID] = vector_1[myID] + vector_2[myID];
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
        hst_vector1[i] = rand() % 10;
        hst_vector2[i] = 0;
    }
    printf("> Vector de %d elementos\n", N);
    printf("> Lanzamiento con %d bloque de %d hilos\n", 1, N);

    // Temporizacion
    // Declaracion de eventos
    cudaEvent_t start;
    cudaEvent_t stop;

    // Creacion de eventos
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // copia de datos hacia el device
    cudaMemcpy(dev_vector1, hst_vector1, N*sizeof(int), cudaMemcpyHostToDevice);

    // marca de inicio
    cudaEventRecord(start, 0);

    // Lanzamos Kernel de un solo bloque y 24 hilos
    suma <<< 1, N >>> (dev_vector1, dev_vector2, dev_resultado, N);

    // marca de final
    cudaEventRecord(stop, 0);

    // recogida de datos desde el device
    cudaMemcpy(hst_vector2, dev_vector2, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hst_resultado, dev_resultado, N*sizeof(int), cudaMemcpyDeviceToHost);

    // sincronizacion GPU-CPU
    cudaEventSynchronize(stop);

    // Calculo del tiempo en milisegundos
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n> Tiempo de ejecucion: %f ms\n", elapsedTime);

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
