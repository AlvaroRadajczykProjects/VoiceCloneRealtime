#include <windows.h>

#include <stdio.h>
#include <stdlib.h>

#include <windows.h>
#include <iostream>
#include <fstream>
#include <string.h>

int getFileSize(FILE* inFile);
FILE* cargarArchivo(const char* filePath);
void crearArchivoEscribirYCerrar(const char* nombre, int nbytes, char* dbytes);
char* leerArchivoYCerrar(const char* nombre, unsigned int* tam);