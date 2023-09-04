#include "WaveFile.h"

using namespace std;

WaveFile::WaveFile(const char* dn) {
	dir_name = dn;
	char* bytes = returnFileData(dir_name, &fileSize);

	memcpy(&ChunkID, bytes+0, 4);
	memcpy(&ChunkSize, bytes+4, 4);
	memcpy(&Format, bytes+8, 4);
	memcpy(&AudioFormat, bytes+20, 2);
	memcpy(&NumChannels, bytes+22, 2);
	memcpy(&SampleRate, bytes+24, 4);
	memcpy(&ByteRate, bytes+28, 4);
	memcpy(&BitsPerSample, bytes+32, 4);

	unsigned long long offset = 12;
	char* NextSubchunkID = new char[5]; NextSubchunkID[4] = NULL;
	memcpy(NextSubchunkID, bytes + offset, 4);
	memcpy(&lenData, bytes + offset + 4, 4);

	while (strcmp(NextSubchunkID, "data") != 0) {
		//cout << "NextSubchunkID: " << NextSubchunkID << endl;
		//cout << "NextSubchunkSize: " << lenData << endl;
		offset += 8 + lenData;
		memcpy(NextSubchunkID, bytes + offset, 4);
		memcpy(&lenData, bytes + offset + 4, 4);
	}

	//cout << "NextSubchunkID: " << NextSubchunkID << endl;
	//cout << "NextSubchunkSize: " << lenData << endl;

	offset += 8;
	data = new char[lenData];
	memcpy(data, bytes+offset, lenData);

	delete bytes;
}

WaveFile::~WaveFile() {
	delete data;
}

void WaveFile::showHeader() {
	cout << endl << endl << "WAVE file header " << dir_name << endl << endl;
	cout << "File size: " << fileSize << endl;
	cout << "ChunkID: "; for(int i = 0; i < 4; i++){ cout << ((char*)&ChunkID)[i]; } cout << endl;
	cout << "ChunkSize: " << ChunkSize << endl;
	cout << "Format: "; for (int i = 0; i < 4; i++) { cout << ((char*)&Format)[i]; } cout << endl;
	cout << "AudioFormat: " << AudioFormat << endl; //3 es coma flotante
	cout << "NumChannels: " << NumChannels << endl;
	cout << "SampleRate: " << SampleRate << endl;
	cout << "ByteRate: " << ByteRate << endl;
	cout << "BitsPerSample: " << BitsPerSample << endl;
	cout << "Data length (bytes): " << lenData << endl;
	cout << endl;
}

const char* WaveFile::getDirName() {
	return dir_name;
}

streampos WaveFile::getfileSize() {
	return fileSize;
}

uint32_t WaveFile::getChunkID() {
	return ChunkID;
}

uint32_t WaveFile::getChunkSize() {
	return ChunkSize;
}

uint32_t WaveFile::getFormat() {
	return Format;
}

uint16_t WaveFile::getAudioFormat() {
	return AudioFormat;
}

uint16_t WaveFile::getNumChannels() {
	return NumChannels;
}

uint32_t WaveFile::getSampleRate() {
	return SampleRate;
}

uint32_t WaveFile::getByteRate() {
	return ByteRate;
}

uint16_t WaveFile::getBlockAlign() {
	return BlockAlign;
}

uint16_t WaveFile::getBitsPerSample() {
	return BitsPerSample;
}

uint32_t WaveFile::getLenData() {
	return lenData;
}

char* WaveFile::getData() {
	return data;
}

