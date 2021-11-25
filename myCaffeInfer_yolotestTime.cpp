#include <unistd.h>
#include <regex>
#include "TrtClassificer.h"
using namespace cv;
using namespace std;


int main(int argc,char ** argv)
{
	cudaSetDevice(0);      //gpuID
    float sumTime = 0;
	 const int INPUT_H = 96; //输入高
    const int INPUT_W = 160;//输入宽
     const int CHANNELS = 3;//通道数
    const char* INPUT_BLOB_NAME = "data";//输入层名

    int MaxBatch=8;

	double allTimeBegin=static_cast<double>(getTickCount());

	vector<vector<float> > meanArray={{ 110.54368197 ,107.80291569 ,107.11519277 },       //pedestrainGlobal mean
		                              { 79.01901062, 78.72796895, 80.79338091 },             //NoStdVehicle    mean
									  { 110.54368197 ,107.80291569 ,107.11519277 }};           ////vehicleSpecial mean
	int AttributeNumber=atoi(argv[6]);
	float mean_data[] = { 79.01901062, 78.72796895, 80.79338091 };//nostd_vehicle
	// float mean_data[] = { 99.95327338, 96.27925874, 86.54154894};//VehicleDriver

	TrtClassificer *pMuclassifier=new TrtClassificer(INPUT_H, INPUT_W, CHANNELS, INPUT_BLOB_NAME,argv[8],argv[9]);
	auto numOfAttributes= pMuclassifier->getNumOfAttribute();
	auto numOfOutputsPerAttrArray =pMuclassifier->numOfOutputsPerAttr();
	vector<string>fileList;
	vector<string> fileType={"jpg"};
    readFileList(argv[4],fileList,fileType);
	 if (strcmp(argv[3],"-s")==0)
	 {
		 pMuclassifier->CaffeToGIEModel(argv[1], argv[2], MaxBatch, argv[5]); 
	 }
	 else if (strcmp(argv[3],"-d")==0)
	 {
		 	int rightLabel = 0;
		
		 Alabel  *pAlabel = new Alabel[numOfOutputsPerAttrArray[AttributeNumber]];
		 for (int i = 0; i < numOfOutputsPerAttrArray[AttributeNumber]; i++)
		 {
			 (pAlabel + i)->pArray = new int[numOfOutputsPerAttrArray[AttributeNumber]];
			 for (int j = 0; j < numOfOutputsPerAttrArray[AttributeNumber]; j++)
			 {
				 pAlabel[i].pArray[j] = 0;
			 }
		 }
		 pMuclassifier->readTrtModel(argv[5]);
		
		 int count = 0;
	
		 float *data = new float[INPUT_H*INPUT_W*CHANNELS];
	   
		 for (auto &file : fileList)
		 {
			 imageProcess((char *)file.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);
			 float **prob3 = new float *[numOfAttributes];
			 for (int i = 0; i < numOfAttributes; i++)
			 {
				 prob3[i] = new float[numOfOutputsPerAttrArray[i]];
			 }
			//  clock_t starts1 = clock();
			 double starts1 = static_cast<double>(getTickCount());
			 pMuclassifier->doInferenceMultiOutPut(data, prob3, 1);  //forward ǰ�����
			//  clock_t ends1 = clock();
			 double ends1 = static_cast<double>(getTickCount());
			 sumTime += (ends1 - starts1)/getTickFrequency()*1000;
			 int max = findMax(prob3[AttributeNumber], numOfOutputsPerAttrArray[AttributeNumber]); //ʶ��ɵı�ǩ
			 int realLabel = getLabel(file); 
			 pAlabel[max].detectLabel++;
			 pAlabel[realLabel].sumLabel++;
			 //pAlabel[realLabel].errorRecognition[max]++;
			 pAlabel[realLabel].pArray[max]++;
            int pos = file.find_last_of("/");
			string picName = file.substr(pos+1);

			 if (max == realLabel)
			 {
				 cout << count++ << " " << picName << " " << (ends1 - starts1)/(getTickFrequency())*1000 <<"  Right"<<endl;
				 pAlabel[max].rightLabel++;
				  rightLabel++;
			 }
			 else
			 {
				 cout << count++ << " " << picName << " " << ends1 - starts1 << "     Error" << endl;
				
				 pAlabel[realLabel].errorLabel++;
				
				 
			 }
			 for (int i = 0; i < numOfAttributes; i++)
			 {
				 delete [] prob3[i];
			 }
			 delete[] prob3;
			
		 }
		//  printf("\n\n%10s %10s %10s %10s %10s %10s\n", "Label", "sumLabels", "rightLabels", "Recall", "detect", "Accuracy");
		//  for (int i = 0; i < numOfOutputsPerAttrArray[AttributeNumber]; i++)
		//  {
		// 	 pAlabel[i].recall = 1.0*pAlabel[i].rightLabel / pAlabel[i].sumLabel;
		// 	 pAlabel[i].accuracy = 1.0*pAlabel[i].rightLabel / pAlabel[i].detectLabel;
		// 	 printf("%10d %10d %10d %10.4f %10d %10.4f\n", i, pAlabel[i].sumLabel, pAlabel[i].rightLabel, pAlabel[i].recall, pAlabel[i].detectLabel, pAlabel[i].accuracy);
		//  }
		//  printf("%10s %10d %10d %10.4f\n\n", "ALL", fileList.size(), rightLabel, 1.0*rightLabel / fileList.size());
		
		//  printf("\nError details:\n");
		//  for (int i = 0; i < numOfOutputsPerAttrArray[AttributeNumber]; i++)
		//  {
		// 	 printf("\n%d recognizedAs :\n", i);
		// 	 for (int j = 0; j < numOfOutputsPerAttrArray[AttributeNumber]; j++)
		// 	 {
		// 		 if(j!=i)
		// 		 printf("%d : %d\t",j, pAlabel[i].pArray[j]);
		// 	 }
		// 	 printf("\n");
			 
		//  }
          cout << "forward All Times: " << sumTime << "ms  Per pic: " << 1.0*sumTime / fileList.size() <<"ms"<< endl;
		//  delete [] feature;
		 delete [] data;
		 for (int i = 0; i < numOfOutputsPerAttrArray[AttributeNumber]; i++)
		 {
			 delete[] pAlabel[i].pArray;
			 //(pAlabel + i)->pArray = new int[attrTagArray[AttributeNumber]];
		 }
		 delete[] pAlabel;
	 }
	 else if(strcmp(argv[3],"-c")==0)    //图片分类 batch=1
	 {
		  pMuclassifier->readTrtModel(argv[5]);
		
		 int count = 0;
		//  float *feature = new float[attrTagArray[0]];
		 float *data = new float[INPUT_H*INPUT_W*CHANNELS];
		 
		 for (auto &file : fileList)
		 {
			 cout<<count++<<" "<<file<<endl;
			//  imageProcess((char *)file.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);

            cv::Mat img = cv::imread(file);
			cv::Mat im;
			cv::resize(img, im, cv::Size(INPUT_W, INPUT_H));

			float *pdata = data;
			setMean(im, mean_data, pdata, CHANNELS, INPUT_H, INPUT_W);

              float **prob3 = new float *[numOfAttributes];
			 for (int i = 0; i < numOfAttributes; i++)
			 {
				 prob3[i] = new float[numOfOutputsPerAttrArray[i]];
			 }
			  double starts1 = static_cast<double>(getTickCount());
			 pMuclassifier->doInferenceMultiOutPut(data, prob3, 1);  //forward ǰ�����
			//  clock_t ends1 = clock();
			 double ends1 = static_cast<double>(getTickCount());
			 sumTime += (ends1 - starts1)/getTickFrequency()*1000;
			 int max = findMax(prob3[AttributeNumber], numOfOutputsPerAttrArray[AttributeNumber]);
			 float  attrProb = prob3[AttributeNumber][max];
            string attrLabel = to_string(max);
			string saveFolder = argv[7]+attrLabel;
			 if (access(saveFolder.c_str(),00))
            {
                  mkdir(saveFolder.c_str(),0777);
            }

            int pos = file.find_last_of("/");
			string picName = file.substr(pos+1);
			string newPicName = to_string(attrProb)+"_"+picName;

			string finalPath = saveFolder+"/"+newPicName;
            
			cv::imwrite(finalPath,img);

			  for (int i = 0; i < numOfAttributes; i++)
			 {
				 delete [] prob3[i];
			 }
			 delete[] prob3;
		 }
    cout << "forward All Times: " << sumTime << "ms  Per pic: " << 1.0*sumTime / fileList.size() <<"ms"<< endl;
		 delete [] data;	
	 }


	  else if(strcmp(argv[3],"-b")==0)    //图片分类  batch>1
	 {
		  pMuclassifier->readTrtModel(argv[5]);
		
		 int count = 0;
		//  float *feature = new float[attrTagArray[0]];
		 float *data = new float[INPUT_H*INPUT_W*CHANNELS*MaxBatch];
		 float *pdata = data;
		 int batchCount= 0;
		 vector<cv::Mat> imgVec(MaxBatch);
		 vector<string> imgName(MaxBatch);
		 for (int ia =0; ia<fileList.size();ia++)
		 {
			 auto file = fileList[ia];
			 cout<<count++<<" "<<file<<endl;
			 
			 cv::Mat img = cv::imread(file);
			 imgVec[batchCount]=img;
			 imgName[batchCount]=file;
			 cv::Mat im;
			 cv::resize(img, im, cv::Size(INPUT_W, INPUT_H));
			 setMean(im, mean_data, pdata, CHANNELS, INPUT_H, INPUT_W);
			 batchCount++;
			float **prob3 = new float *[numOfAttributes];
			 for (int i = 0; i < numOfAttributes; i++)
			 {
				 prob3[i] = new float[numOfOutputsPerAttrArray[i]*MaxBatch];
			 }
			 if (batchCount%MaxBatch==0 || ia==fileList.size()-1)
			 {
				 double starts1 = static_cast<double>(getTickCount());
	             pMuclassifier->doInferenceMultiOutPut(data, prob3, batchCount);
				 double ends1 = static_cast<double>(getTickCount());
				 sumTime += (ends1 - starts1)/getTickFrequency()*1000;
				 for(int attr = 0; attr<batchCount; attr++)
				 {
				 int predictLabel = findMax(prob3[AttributeNumber]+attr*numOfOutputsPerAttrArray[AttributeNumber], numOfOutputsPerAttrArray[AttributeNumber]);
				 float predictProb =(prob3[AttributeNumber]+attr*numOfOutputsPerAttrArray[AttributeNumber])[predictLabel];
				//  cout<<predictProb<<" "<<predictLabel<<" "<<imgName[attr]<<endl;
				string attrLabel = to_string(predictLabel);
				string saveFolder = argv[7]+attrLabel;
				if (access(saveFolder.c_str(),00))
				{
				mkdir(saveFolder.c_str(),0777);
				}

				int pos = imgName[attr].find_last_of("/");
				string picName =  imgName[attr].substr(pos+1);
				string newPicName = to_string(predictProb)+"_"+picName;

				string finalPath = saveFolder+"/"+newPicName;

				cv::imwrite(finalPath,imgVec[attr]);
				 }

            


				 batchCount=0;
				 pdata=data;
			 }
			  for (int i = 0; i < numOfAttributes; i++)
			 {
				 delete [] prob3[i];
			 }
			 delete[] prob3;
			
		 }
         cout << "forward All Times: " << sumTime << "ms  Per pic: " << 1.0*sumTime / fileList.size() <<"ms"<< endl;
             delete [] data;
	 }
	 cout<<endl;
	 delete pMuclassifier;
	 double allTimeEnd = static_cast<double>(getTickCount());;
	 double timeGap = allTimeEnd - allTimeBegin;
	 double fruency = getTickFrequency();
	 double useTime = (allTimeEnd - allTimeBegin) / getTickFrequency();
	 cout<<"done"<<endl;
	 cout << "Total program time:" << useTime << " s"<<endl;
	 cout<<"done"<<endl;
	return 0;
}


// int main_plate(int argc,char ** argv)
// {
// 	cudaSetDevice(0);
// 	double allTimeBegin=static_cast<double>(getTickCount());
// 	int rightLabel = 0;
// 	int errorLabel = 0;
// 	int sumRightlabel = 0;
// 	// int AttributeNumber = 1;  //�������
// 	float sumTime = 0;
// 	bool saveStat = false;
// 	char *saveMode ="-d";
// 	int count13 = 0;
// 	int tabelsnum = 3622;
// 	int time_step = 21;
// 	int alphabet_size = tabelsnum;
// 	// char *attributeNumber="0";
// 	//int numOfAttribute = 11; //numOfAttributes
// 	//  char *modelFile = "/home/cxl/tensorCaffe/AttributeTest/carPlate/squeeze-48x168_iter_38000.caffemodel";
// 	//  char *deployFile = "/home/cxl/tensorCaffe/AttributeTest/carPlate/deploy_resnet_noblstm_deep_384_conv_78_chinese3.prototxt";
// 	// argv[1]=deployFile;
// 	// argv[2]=modelFile;
// 	strcpy(argv[3],saveMode);
// 	std::vector<std::string> outPutNameArray{"fc1x2"};  //����������
// 	//int attrTagArray[11];  //ÿ�����Ե�������С
// 	int numOfAttribute = 1;
// 	int *attrTagArray = new int[numOfAttribute];
// 	attrTagArray[0]=tabelsnum*time_step;
// 	// getNumOfAttribute(string(argv[1]), attrTagArray);
// 	// auto numOfAttribute=getNameAndNumber(string(argv[1]), outPutNameArray, attrTagArray);   //���ÿ�����Ե���������ÿ���������������� ��prob_1,prob_2�ȵ�;
// 	// char *filePath = "/home/cxl/data/huolala";   //����ȫ�������ļ���
// 	// argv[4]=filePath;
// 	// char *filePath = "/home/cxl/tensorCaffe/AttributeTest/testData";
// 	// char *trtSavePath = "/home/cxl/tensorCaffe/AttributeTest/build/carPlate.engine";
// 	// argv[5]=trtSavePath;
// 	// float mean_data[] = { 0, 0, 0 }; //pedestrain_global
// 	int AttributeNumber=atoi(argv[6]);
// 	//float mean_data[] = { 110.54368197 ,107.80291569 ,107.11519277 }; //vehicleSpecial mean
// 	float mean_data[] = { 79.01901062, 78.72796895, 80.79338091 };//nostd_vehicle
// 	TrtClassificer *pMuclassifier=new TrtClassificer(INPUT_H, INPUT_W, CHANNELS, INPUT_BLOB_NAME,argv[8],argv[9]);
	
// 	vector<string>fileList;
// 	vector<string> fileType={"jpg"};
// 	// vector<int>label;
//     readFileList(argv[4],fileList,fileType);
// 	// readDir(filePath, fileList, label);
// 	 if (strcmp(argv[3],"-s")==0)
// 	 {
// 		 pMuclassifier->CaffeToGIEModel(argv[1], argv[2], outPutNameArray, 1, argv[5]); 
// 	 }
// 	 else {
// 		// int time_step = 21;
// 		// int alphabet_size = tabelsnum;
// 		float * prob2 = new float[attrTagArray[0]];
// 		pMuclassifier->readTrtModel(argv[5]);

// 		int count = 0;
// 		float *feature = new float[attrTagArray[0]];
// 		float *data = new float[INPUT_H*INPUT_W*CHANNELS];

// 		for (auto &file : fileList)
// 		{
// 			imageProcess((char *)file.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);//Ԥ���� resize��ȥ��ֵ
// 			float **prob3 = new float *[numOfAttribute];
// 			for (int i = 0; i < numOfAttribute; i++)
// 			{
// 				prob3[i] = new float[attrTagArray[i]];
// 			}
// 			auto starts1 = cv::getTickCount();
// 			pMuclassifier->doInferenceMultiOutPut(data, prob3, 1);  //forward ǰ�����
// 			auto ends1 = cv::getTickCount();
// 			sumTime += (ends1 - starts1);
// 			cout << (ends1 - starts1)/cv::getTickFrequency()*1000 << endl;
// 			int k = 0;
// 			for (int i = 0; i < time_step; i++)
// 			{
// 				for (int j = 0; j < alphabet_size; j++)
// 					prob2[k++] = prob3[0][i + j * time_step];
// 			}
// 			int blank_label = 0;
// 			int prev_label = blank_label;
// 			string result;
// 			vector<string> labelMap = getLabelChinese();

// 			for (int i = 0; i < time_step; ++i)
// 			{
// 				float* lin = prob2 + i * tabelsnum;
// 				int predict_label = std::max_element(lin, lin + tabelsnum) - lin;
// 				//cout << "predict_label: " << predict_label << endl;

// 				if (predict_label != blank_label && predict_label != prev_label)
// 				{
// 					/*float pro = 1 / (1 + exp(-(lin[predict_label] - 10.2) / 4));
// 					pro_temp.push_back(pro);
// 					if (pro < min)
// 					{
// 					min = pro;
// 					}*/
// 					result = result + getLabel1(labelMap, predict_label);
// 				}

// 				prev_label = predict_label;
// 			}
// 			cout << result << endl;
// 			for (int i = 0; i < numOfAttribute; i++)
// 			{
// 				delete[] prob3[i];
// 			}
// 			delete[] prob3;

// 		}
		
// 		delete[] feature;
// 		delete[] data;
// 		delete[] prob2;
// 	 }
// 	 cout<<endl;
// 	 delete pMuclassifier;
// 	 delete[] attrTagArray;
// 	 double allTimeEnd = static_cast<double>(getTickCount());;
// 	 double timeGap = allTimeEnd - allTimeBegin;
// 	 double fruency = getTickFrequency();
// 	 double useTime = (allTimeEnd - allTimeBegin) / getTickFrequency();
// 	 cout<<endl;
// 	 cout << "Total program time:" << useTime << " s"<<endl;
// 	//system("pause");
// 	return 0;
// }