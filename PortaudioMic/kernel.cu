#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "portaudio.h"

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 256 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil

using namespace std;

static int patestCallback(
	const void* inputBuffer,
	void* outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void* userData
) {

	float* ve = (float*)inputBuffer;
	float* vs = (float*)outputBuffer;

	//for (int i = 0; i < FRAMES_PER_BUFFER; i++) { vs[i] = (i + 1) / (float)FRAMES_PER_BUFFER; }

	memcpy(vs, ve, FRAMES_PER_BUFFER * sizeof(float));

	return paContinue;
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

void showDevices(int numDevices) {
	const PaDeviceInfo* deviceInfo;
	for (int i = 0; i < numDevices; i++) {
		deviceInfo = Pa_GetDeviceInfo(i);
		printf("Device: %d\n", i);
		printf("    name: %s\n", deviceInfo->name);
		printf("    maxInputChannels: %d\n", deviceInfo->maxInputChannels);
		printf("    maxOutputChannels: %d\n", deviceInfo->maxOutputChannels);
		printf("    defaultSampleRate: %f\n", deviceInfo->defaultSampleRate);
	}
}

int main() {

	PaError err;
	err = Pa_Initialize();
	checkErr(err);

	int numDevices = Pa_GetDeviceCount();
	manageNumberDevices(numDevices);
	showDevices(numDevices);

	//recuerda en el VAC control panel de que hayan dos canales, y de reiniciar lo de trial, que sino da por saquito...
	int input_device = 39; //linea 1 de entrada (name: Line 1 (Virtual Cable 1) maxInputChannels: 2 maxOutputChannels: 0), debe de iniciarse el audio repeater con entrada micrófono y salida la linea 1
	int output_device = 45; //linea 2 de salida (name: Line Out (Virtual Cable 2) maxInputChannels: 0 maxOutputChannels: 2)

	PaStreamParameters inputParameters;
	PaStreamParameters outputParameters;

	memset(&inputParameters, 0, sizeof(inputParameters));
	inputParameters.channelCount = 1;
	inputParameters.device = input_device;
	inputParameters.hostApiSpecificStreamInfo = NULL;
	inputParameters.sampleFormat = paFloat32;
	inputParameters.suggestedLatency = Pa_GetDeviceInfo(input_device)->defaultLowInputLatency;

	memset(&outputParameters, 0, sizeof(outputParameters));
	outputParameters.channelCount = 1;
	outputParameters.device = output_device;
	outputParameters.hostApiSpecificStreamInfo = NULL;
	outputParameters.sampleFormat = paFloat32;
	outputParameters.suggestedLatency = Pa_GetDeviceInfo(input_device)->defaultLowOutputLatency;

	PaStream* stream;
	err = Pa_OpenStream(
		&stream,
		&inputParameters,
		&outputParameters,//&outputParameters,
		SAMPLE_RATE,
		FRAMES_PER_BUFFER,
		paNoFlag,
		patestCallback,
		NULL
	);
	checkErr(err);

	/*
	PaStream* stream;
	err = Pa_OpenDefaultStream(&stream, 1, 1, paFloat32, SAMPLE_RATE, FRAMES_PER_BUFFER, patestCallback, nullptr);
	checkErr(err);
	*/

	err = Pa_StartStream(stream);
	checkErr(err);

	Pa_Sleep(100 * 1000);

	err = Pa_StopStream(stream);
	checkErr(err);

	err = Pa_CloseStream(stream);
	checkErr(err);

	err = Pa_Terminate();
	checkErr(err);

	return EXIT_SUCCESS;
}