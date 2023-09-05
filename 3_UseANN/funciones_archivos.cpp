#include "funciones_archivos.h"

using namespace std;
namespace fs = std::filesystem;

char* returnFileData(const char* name, streampos* size) {
    char* memblock;
    ifstream file(name, ios::in | ios::binary | ios::ate);
    if (file.is_open()) {
        *size = file.tellg();
        memblock = new char[*size];
        file.seekg(0, ios::beg);
        file.read(memblock, *size);
        file.close();
        return memblock;
    }
    return NULL;
}

std::vector<string> getVectorCharArrayFilesInDirectory(const char* directorio) {
    fs::path p(directorio);
    std::vector<string> s;

    for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); i++)
    {
        std::string cadenastr = i->path().filename().string().c_str();
        s.push_back(cadenastr);
    }

    return s;
}

std::vector<string> intersection(std::vector<string> v1, std::vector<string> v2) {
    std::vector<string> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(), v1.end(),
        v2.begin(), v2.end(),
        back_inserter(v3));
    return v3;
}

void showVectorCharArray(std::vector<string> s) {
    for (std::vector<string>::iterator it = s.begin(); it != s.end(); ++it) {
        std::string cadena = *it;
        std::cout << ' ' << cadena;
    }
    std::cout << '\n';
}

char* stdStringAPunteroChar(string s) {
    char* p = (char*)malloc((s.length() + 1) * sizeof(char));
    p[s.length()] = NULL;
    strcpy(p, s.c_str());
    return p;
}

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