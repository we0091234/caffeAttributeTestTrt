#ifndef TRTCLASSFIER_H_
#define TRTCLASSFIER_H_

// #include "logger.h"
#include "logging.h"
// #include "common.h"
// #include "argsParser.h"
// #include "buffers.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include<time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include<windows.h>
#include<string>
#include <iomanip>
#include <vector>
#include "utils.h"
static sample::Logger gLogger;
using namespace nvinfer1;
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

class TrtClassificer
{
public:
	TrtClassificer(int INPUT_H,int INPUT_W,int CHANNELS,const char * INPUT_NAME,const char *OUTPUT_NAME,int outputSize);
	TrtClassificer(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME, std::vector<std::string>OUTPUT_NAME, int outputSize);
    TrtClassificer(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME,const char * outPutNameArray,const char *everyAttrNumArray);
	// void doInferenceMultiOutPut(float* input, float **&output, int batchSize);
	
    int getNumOfAttribute();
    int * numOfOutputsPerAttr();
    void imageProcess(char *picName, float *data,float *mean_data);

	void CaffeToGIEModel(const char* deployFile,const char* modelFile, unsigned int maxBatchSize, const char * TrtSaveFileName);
	void doInference( float* input, float* output, int batchSize);
	void doInference(IExecutionContext& context, float* input, float **&output, int batchSize);
	void doInferenceMultiOutPut(float* input, float **&output, int batchSize);
	void saveToTrtModel(const char * TrtSaveFileName)
	{
		std::ofstream out(TrtSaveFileName, std::ios::binary);
		if (!out.is_open())
		{
		std::cout << "打开文件失败!" <<std:: endl;
		}
		out.write(reinterpret_cast<const char*>(this->_gieModelStream->data()), this->_gieModelStream->size());
		out.close();
	}
	void  readTrtModel(const char * Trtmodel)
	{
		size_t size{ 0 };
		std::ifstream file(Trtmodel, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			_trtModelStream = new char[size];
			assert(_trtModelStream);
			file.read(_trtModelStream, size);
			file.close();
		}
		_trtModelStreamSize = size;

		_runtime = createInferRuntime(gLogger);
		_engine1 = _runtime->deserializeCudaEngine(_trtModelStream, _trtModelStreamSize);
		//cudaSetDevice(0);
		context = _engine1->createExecutionContext();
		std::cout<<"woshirencai"<<std::endl;
	}
	
	//void mydoInference(IExecutionContext& context, float* input, float *output, int batchSize);

	 virtual ~TrtClassificer()
	{
		free(_inputName);
		free(_outputName);
		if (_gieModelStream)
			_gieModelStream->destroy();
		delete _trtModelStream;
		if(context)
		context->destroy();
		if(_engine1)
		_engine1->destroy();
		if (_runtime)
		_runtime->destroy();
		if(m_numOutPut)
		delete [] m_numOutPut;
		std::cout << "Trt destory is running" << std::endl;

	}
	protected:
	int _input_h;
	int _input_w;
	int _channel;
	int _batchSize;
	IHostMemory *_gieModelStream{ nullptr };
	char *_inputName{nullptr};
	char *_outputName{nullptr};
	int  _outputNumber;
	char *_trtModelStream{ nullptr };
	int _trtModelStreamSize;
	IRuntime* _runtime{nullptr};
	ICudaEngine* _engine1{ nullptr };
	IExecutionContext *context{nullptr};

	int m_numAttribute{0};
	int *m_numOutPut{nullptr};

	float *data{nullptr};
	
	std::vector<std::string> m_outPutName;
};
#endif // !1