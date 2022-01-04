#include "TrtClassificer.h"
#include "calibrator.h"

using namespace nvcaffeparser1;
// #define numAttribute 11
// #define USE_INT8

//static sample::Logger gLogger;
TrtClassificer::TrtClassificer(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME, const char *OUTPUT_NAME, int outputSize)
{
	this->_input_h = INPUT_H;
	this->_input_w = INPUT_W;
	this->_channel = CHANNELS;
	this->_inputName = strdup(INPUT_NAME);
	this->_outputName = strdup(OUTPUT_NAME);
	this->_outputNumber = outputSize;

}


 TrtClassificer::TrtClassificer(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME,const char * outPutNameString,const char *everyAttrNumArray)
 {
    this->_input_h = INPUT_H;
	this->_input_w = INPUT_W;
	this->_channel = CHANNELS;
	this->_inputName = strdup(INPUT_NAME);
	this->m_outPutName=split(outPutNameString,",");
	this->m_numAttribute = this->m_outPutName.size();
    auto  numArray = split(everyAttrNumArray,",");
	m_numOutPut = new int[this->m_numAttribute];
	for(int i = 0; i<this->m_numAttribute ;i++)
     this->m_numOutPut[i] =atoi(numArray[i].c_str());
 }


void TrtClassificer::CaffeToGIEModel(const char* deployFile, const char* modelFile, unsigned int maxBatchSize, const char * TrtSaveFileName)
{

	std::cout << "Convert Caffemodel to  Trt  model...." << std:: endl;
	IBuilder* builder = createInferBuilder(gLogger);

	IBuilderConfig *config= builder->createBuilderConfig();

	INetworkDefinition* network = builder->createNetworkV2(0U);
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile, modelFile, *network, nvinfer1::DataType::kFLOAT);
	for (auto& s : this->m_outPutName)
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 30); 

#ifdef USE_FP16
	config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
	std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
	assert(builder->platformHasFastInt8());
	config->setFlag(BuilderFlag::kINT8);
	Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, this->_input_w, this->_input_h, "/home/data/cxl/calib/", "int8calib.table", this->_inputName);
	config->setInt8Calibrator(calibrator);
#endif
	// ICudaEngine* engine = builder->buildCudaEngine(*network);
	ICudaEngine* engine =builder->buildEngineWithConfig(*network, *config);
	assert(engine);
	network->destroy();
	parser->destroy();
	this->_gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	config->destroy();
	shutdownProtobufLibrary();
	this->saveToTrtModel(TrtSaveFileName);
	std::cout << "Convert Done!" << std::endl;
}

void TrtClassificer::doInference(float* input, float* output, int batchSize)
{
	
	const ICudaEngine& engine= (*context).getEngine();
	assert(engine.getNbBindings() == 2);
	void* buffers[2];
	int inputIndex = engine.getBindingIndex(this->_inputName),
		outputIndex = engine.getBindingIndex(this->_outputName);
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * this->_channel * this->_input_h * this->_input_w * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * this->_outputNumber * sizeof(float)));
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * this->_channel * this->_input_h * this->_input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
	(*context).enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize *this->_outputNumber * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


void TrtClassificer::doInferenceMultiOutPut(float* input, float **&output, int batchSize)
{
	//int numAttribute = 11;
	// std::vector<std::string> haha = { "prob_1","prob_2","prob_3", "prob_4" ,"prob_5" ,"prob_6" ,"prob_7" ,"prob_8" ,"prob_9" ,"prob_a" ,"prob_b" };
	//int numAttri[] = { 2,11,11,2,3,2,4,3,4,2,4 };
	const ICudaEngine& engine = (*context).getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	// std::cout<<engine.getNbBindings()<<std::endl;
	// std::cout<<m_numAttribute+1<<std::endl;
	assert(engine.getNbBindings() == m_numAttribute+1); //numofAttribute  and   input  layer
	void* buffers[12];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(this->_inputName);
	/*outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME),
	outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);*/
	int opIndex[11];
	for (int i = 0; i < m_numAttribute; i++)
	{
		opIndex[i] = engine.getBindingIndex(m_outPutName[i].c_str());
	}

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize *this->_channel *this->_input_h * this->_input_w * sizeof(float)));
	for (int i = 0; i < m_numAttribute; i++)
	{
		CHECK(cudaMalloc(&buffers[opIndex[i]], batchSize * m_numOutPut[i] * sizeof(float)));
	}
	/*CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));*/
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * this->_channel *this->_input_h * this->_input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
	(*context).enqueue(batchSize, buffers, stream, nullptr);

	for (int i = 0; i < m_numAttribute; i++)
	{
		CHECK(cudaMemcpyAsync(output[i], buffers[opIndex[i]], batchSize *m_numOutPut[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));
	}
	/*CHECK(cudaMemcpyAsync(output[0], buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(output[1], buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));*/
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	for (int i = 0; i < m_numAttribute; i++)
	{
		/*CHECK(cudaMemcpyAsync(output[i], buffers[opIndex[i]], batchSize *numAttri[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));*/
		CHECK(cudaFree(buffers[opIndex[i]]));
	}
}

int TrtClassificer:: getNumOfAttribute()
{
	return this->m_numAttribute;
}

  int *TrtClassificer:: numOfOutputsPerAttr()
  {
	  return this->m_numOutPut;
  }