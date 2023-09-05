#include "funciones_archivos.h"

using namespace std;

class WaveFile {

	private:
		const char* dir_name;
		streampos fileSize;
		uint32_t ChunkID;				//4 big
		uint32_t ChunkSize;				//4 little
		uint32_t Format;				//4 big
		uint16_t AudioFormat;			//2 little
		uint16_t NumChannels;			//2 little
		uint32_t SampleRate;			//4 little
		uint32_t ByteRate;				//4 little
		uint16_t BlockAlign;			//2 little
		uint16_t BitsPerSample;			//2 little
		uint32_t lenData;
		char* data;					//Subchunk2Size little

	public:
		WaveFile(const char* dn);
		~WaveFile();
		void showHeader();
		const char* getDirName();
		streampos getfileSize();
		uint32_t getChunkID();
		uint32_t getChunkSize();
		uint32_t getFormat();
		uint16_t getAudioFormat();
		uint16_t getNumChannels();
		uint32_t getSampleRate();
		uint32_t getByteRate();
		uint16_t getBlockAlign();
		uint16_t getBitsPerSample();
		uint32_t getLenData();
		char* getData();

};

