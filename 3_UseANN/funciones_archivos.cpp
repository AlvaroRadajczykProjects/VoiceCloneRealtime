#include "funciones_archivos.h"

int getFileSize(FILE* inFile)
{
    int fileSize = 0;
    fseek(inFile, 0, SEEK_END);

    fileSize = ftell(inFile);

    fseek(inFile, 0, SEEK_SET);
    return fileSize;
}

FILE* cargarArchivo(const char* filePath) {
    FILE* file = fopen(filePath, "r");
    if (file == nullptr)
    {
        fprintf(stderr, "Unable to open file: %s\n", filePath);
        exit(EXIT_FAILURE);
    }
    return file;
}

void crearArchivoEscribirYCerrar(const char* nombre, int nbytes, char* dbytes) {
    FILE* f = fopen(nombre, "wb");
    fwrite(dbytes, 1, nbytes, f);
    fclose(f);
}

//en tam, poner la dirección de memoria de un entero
char* leerArchivoYCerrar(const char* nombre, unsigned int* tam) {
    FILE* f = fopen(nombre, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* string = (char*)malloc(fsize);
    fread(string, 1, fsize, f);
    fclose(f);

    return string;
}