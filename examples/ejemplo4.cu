// Hilos y Bloques
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define COLUMNAS 10     // Nro de columnas -> eje x
#define FILAS 6         // Nro de filas -> eje y

// Kernel Bidimensional (x, y)
__global__
void MathFinal(int *entrada, int *salida)
{
    // indice de la columna: eje x
    int columna = threadIdx.x;

    // indice de la fila: eje y
    int fila = threadIdx.y;

    // Kernel de un solo bloque:
    // indice lineal
    int globalID = columna + fila * COLUMNAS;

    // Indice Lineal Traspuesto
    int idTrasp = fila + columna * FILAS;

    // Escritura en la matriz final
    salida[idTrasp] = entrada[globalID];
}

int main(int argc, char** argv) {
    // declaraciones
    int *hst_Entrada, *hst_Salida;
    int *dev_Entrada, *dev_Salida;

    // reserva en el host
    hst_Entrada = (int*)malloc(FILAS*COLUMNAS * sizeof(int));
    hst_Salida = (int*)malloc(FILAS*COLUMNAS * sizeof(int));

    // reserva en el device
    cudaMalloc((void**)&dev_Entrada, FILAS*COLUMNAS * sizeof(int));
    cudaMalloc((void**)&dev_Salida, FILAS*COLUMNAS * sizeof(int));

    // inicializacion
    for (int i=0; i<(FILAS*COLUMNAS); i++) {
        hst_Entrada[i] = i + 1; // numeros secuenciales desde el 1
        hst_Salida[i] = 0;
    }

    // dimensiones del kernel
    // 1 Bloque
    dim3 Nbloques(1);

    // bloque bidimensional (x,y)
    // Eje x-> COLUMNAS
    // Eje y-> FILAS
    dim3 hilosB(COLUMNAS, FILAS);

    // copia de datos hacia el device
    cudaMemcpy(dev_Entrada, hst_Entrada, FILAS*COLUMNAS * sizeof(int), cudaMemcpyHostToDevice);

    // Numero de hilos
    printf("> KERNEL de 1 BLOQUE con %d HILOS:\n", COLUMNAS*FILAS);
    printf("  eje x -> %2d hilos\n  eje y -> %2d hilos\n", COLUMNAS, FILAS);

    // Lanzamiento del Kernel
    MathFinal <<< Nbloques, hilosB >>>(dev_Entrada, dev_Salida);

    // recogida de datos desde el device
    cudaMemcpy(hst_Salida, dev_Salida, FILAS*COLUMNAS * sizeof(int), cudaMemcpyDeviceToHost);


    // impresion de resultados
    printf("> MATRIZ ORIGINAL:\n");
    for (int i = 0; i<FILAS; i++) {
        for (int j = 0; j<COLUMNAS; j++) {
            printf("%3d ", hst_Entrada[j + i*COLUMNAS]);
        }
        printf("\n");
    }
    printf("\n");

    printf("> MATRIZ FINAL:\n");
    for (int i = 0; i<COLUMNAS; i++) {
        for (int j = 0; j<FILAS; j++) {
            printf("%3d ", hst_Salida[j + i*FILAS]);
        }
        printf("\n");
    }

    // salida del programa
    printf("\n<pulsa [INTRO] para finalizar>\n");
    getchar();
    return 0;
}
