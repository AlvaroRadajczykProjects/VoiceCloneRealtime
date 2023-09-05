//#include "funciones_archivos.h"
#include "WaveFile.h"
#include "Network.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include <portaudio.h>

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 512 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil


struct pointers {
    float* ptr1;
    float* ptr2;
};

struct pointers devolverDatosEntrenamiento(int* numero_ejemplos, unsigned long long* tam_arr) {

    struct pointers p;

    std::vector<string> archivos = intersection(
        getVectorCharArrayFilesInDirectory("..\\audio_source"),
        getVectorCharArrayFilesInDirectory("..\\audio_target")
    );

    *numero_ejemplos = archivos.size();

    char* narchivo;
    unsigned long long ttotal_entrada = 0;
    unsigned long long ttotal_salida = 0;

    unsigned long long offset_entrada = 0;
    unsigned long long offset_salida = 0;

    int tbytes_segmento = FRAMES_PER_BUFFER * 4;

    //tomar el tamaño total del array a crear
    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {

        string cadena_completa = "..\\audio_source\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f = new WaveFile(narchivo);

        free(narchivo);
        cadena_completa = "..\\audio_target\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f2 = new WaveFile(narchivo);

        //calcular
        ttotal_entrada += f->getLenData() + (tbytes_segmento - (f->getLenData() % tbytes_segmento));
        ttotal_salida += f2->getLenData() + (tbytes_segmento - (f2->getLenData() % tbytes_segmento));

        if (ttotal_entrada != ttotal_salida) {
            printf("\nError al cargar datos: tienen un tiempo de duracion distintos los archivos con el nombre %s\n", narchivo);
            exit(EXIT_FAILURE);
        }

        free(narchivo);

        delete f;
        delete f2;
    }

    *tam_arr = ttotal_entrada / 4;

    p.ptr1 = new float[ttotal_entrada];
    p.ptr2 = new float[ttotal_salida];

    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {
        string cadena_completa = "..\\audio_source\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f = new WaveFile(narchivo);

        free(narchivo);
        cadena_completa = "..\\audio_target\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f2 = new WaveFile(narchivo);

        free(narchivo);

        //pasar datos
        memcpy(p.ptr1 + offset_entrada, (float*)f->getData(), f->getLenData());
        offset_entrada += f->getLenData() / 4;
        while (offset_entrada % FRAMES_PER_BUFFER != 0) {
            p.ptr1[offset_entrada] = 0.0;
            offset_entrada++;
        }

        memcpy(p.ptr2 + offset_salida, (float*)f2->getData(), f2->getLenData());
        offset_salida += f2->getLenData() / 4;
        while (offset_salida % FRAMES_PER_BUFFER != 0) {
            p.ptr2[offset_salida] = 0.0;
            offset_salida++;
        }

        delete f;
        delete f2;
    }

    return p;

}

static void checkErr(PaError err) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

void manageNumberDevices(int numDevices) {
    printf("Number of devices: %d\n", numDevices);
    if (numDevices < 0) {
        printf("Error getting device count.\n");
        exit(EXIT_FAILURE);
    }
    else if (numDevices < 1) {
        printf("No record devices available.\n");
        exit(EXIT_FAILURE);
    }
}

extern __device__ func_t d_ELU;
extern __device__ func_t d_dELU;
extern __device__ func_t d_Linear;
extern __device__ func_t d_dLinear;
extern __device__ func2_t d_MSE;
extern __device__ func2_t d_dMSE;

func_t getDeviceSymbolInGlobalMemory(func_t d_arrfunc) {
    func_t h_arrfunc;
    cudaMemcpy(&h_arrfunc, d_arrfunc, sizeof(func_t), cudaMemcpyDeviceToHost);
    return h_arrfunc;
}

func2_t getDeviceSymbolInGlobalMemory(func2_t d_func) {
    func2_t h_func;
    cudaMemcpy(&h_func, d_func, sizeof(func2_t), cudaMemcpyDeviceToHost);
    return h_func;
}

func_t d_func = 0;
func2_t d_func2 = 0;
func3_t d_func3 = 0;

float val = 0.04;

//first epoch number is 0
float lrate_func(int epoch) {
    if (epoch % 2000 == 0) { val = val / (float)2; }
    return val;
}

int main() {

    srand(time(NULL));

    /*

        el formato de los archivos de audio será wave pista mono 32-bit float point (coma flotante), una frecuencia de 44100Hz,
        con la cabecera totalmente limpia (según documentaciones oficiales, de tamaño 44 bytes)

        si quieres ajustar un audio de 48000Hz a 44100Hz, en audacity hay que seleccionar toda la pista, efecto->tono y tempo->
        cambiar la velocidad y cambiar el factor multiplicador por 1,088. Lo exportas y listo. No merece la pena cambiar
        la frecuencia desde audacity, lo hace mal, y luego no se guarda, para eso habría que tocar la cabecera

        en la carpeta audio_source, se deben guardar los audios de voz de la voz de la persona que quiere entrenal el modelo

        en la carpeta audio_target, se deben guardar los audios equivalentes de audio_source pero de la voz que se quiere clonar

        los archivos de audio que tengan el mismo contenido pero distinta voz deben de tener el mismo nombre. Además, deben tener
        el mismo tamaño, sino se entrenará a la red con los datos del más pequeño (es decir, si uno dura 3s y otro 4s, se toma el
        ejemplo  como los 3s enteros del primer audio, y los 3 primeros segundos del segundo audio, se le ha truncado 1s del final

        También, deben haber el mismo número de archivos en ambas carpetas, sino se ignorarán los que no tengan su correspondiente
        archivo pareja con el mimo nombre
    */

    cudaMalloc(&d_func, sizeof(func_t));

    cudaGetSymbolAddress((void**)&d_func, d_ELU);
    func_t ELU = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_dELU);
    func_t dELU = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_Linear);
    func_t Linear = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func, d_dLinear);
    func_t dLinear = getDeviceSymbolInGlobalMemory(d_func);

    cudaGetSymbolAddress((void**)&d_func2, d_MSE);
    func2_t MSE = getDeviceSymbolInGlobalMemory(d_func2);

    cudaGetSymbolAddress((void**)&d_func2, d_dMSE);
    func2_t dMSE = getDeviceSymbolInGlobalMemory(d_func2);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int numero_ejemplos = 0;
    unsigned long long tam_arr = 0;
    float* entrada = 0;
    float* salida = 0;

    struct pointers p;
    p = devolverDatosEntrenamiento(&numero_ejemplos, &tam_arr);

    entrada = p.ptr1;
    salida = p.ptr2;

    const int nejemplos = tam_arr / FRAMES_PER_BUFFER;//tam_arr/1024;//tam_arr%1024;
    int nejemplos_train = (int)nejemplos * 0.8;
    int nejemplos_test = nejemplos - nejemplos_train;

    printf("\nNumero de ejemplos: %d", nejemplos);
    printf("\nNumero de ejemplos train: %d", nejemplos_train);
    printf("\nNumero de ejemplos test: %d\n", nejemplos_test);

    //for (int i = 0; i < tam_arr; i++) { printf("%f, ", entrada[i]); }
    //printf("\nCaca: %d", tam_arr% FRAMES_PER_BUFFER);

    Network* n = new Network(FRAMES_PER_BUFFER, 1, 3, new Layer * [3] {
        new Layer(FRAMES_PER_BUFFER, ELU, dELU),
        new Layer(FRAMES_PER_BUFFER, ELU, dELU),
        new Layer(FRAMES_PER_BUFFER, Linear, dLinear),
    }, MSE, dMSE, true);

    n->loadNetworkFromFile("..\\network_trained_bias_weights.data");

    n->initForward(1);

    PaError err;
    err = Pa_Initialize();
    checkErr(err);

    int numDevices = Pa_GetDeviceCount();
    manageNumberDevices(numDevices);

    PaStream* stream;
    err = Pa_OpenDefaultStream(&stream, 0, 1, paFloat32, SAMPLE_RATE, FRAMES_PER_BUFFER, nullptr, nullptr);
    checkErr(err);

    err = Pa_StartStream(stream);
    checkErr(err);

    float* buffer = new float[FRAMES_PER_BUFFER];

    printf("\n\nHearing training data");

    for (int i = 0; i < nejemplos_train * FRAMES_PER_BUFFER; i += FRAMES_PER_BUFFER) {
        n->forward(1, entrada + i, buffer);
        err = Pa_WriteStream(stream, buffer, FRAMES_PER_BUFFER);

        //err = Pa_WriteStream(stream, entrada + i, FRAMES_PER_BUFFER);

        checkErr(err);
    }

    printf("\n\nWait hear with validation data");

    err = Pa_StopStream(stream);
    checkErr(err);

    Sleep(2000);

    err = Pa_StartStream(stream);
    checkErr(err);

    printf("\n\nHearing validation data");

    for (int i = nejemplos_train * FRAMES_PER_BUFFER; i < nejemplos * FRAMES_PER_BUFFER; i += FRAMES_PER_BUFFER) {
        n->forward(1, entrada + i, buffer);
        err = Pa_WriteStream(stream, buffer, FRAMES_PER_BUFFER);

        //err = Pa_WriteStream(stream, entrada + i, FRAMES_PER_BUFFER);

        checkErr(err);
    }

    err = Pa_StopStream(stream);
    checkErr(err);

    err = Pa_CloseStream(stream);
    checkErr(err);

    err = Pa_Terminate();
    checkErr(err);

    n->finalizeForwardBackward();

    delete n;

    return 0;
}