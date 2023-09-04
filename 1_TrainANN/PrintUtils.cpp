#include "PrintUtils.h"

void imprimirVectorPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
    printf("\n%s [ ", texto_mostrar);
    for (int i = inicio; i < fin; i++) {
        printf("%.20f", vector[i]);
        if (i < fin - 1) { printf(","); }
        printf(" ");
    }
    printf("]");
}

void imprimirMatrizPorPantalla(char* texto_mostrar, float matriz[], int n_filas, int n_columnas) {
    printf("\n%s\n", texto_mostrar);
    for (int i = 0; i < n_filas; i++) {
        imprimirVectorPorPantalla(" ", matriz, i * n_columnas, i * n_columnas + n_columnas);
    }
    printf("\n");
}