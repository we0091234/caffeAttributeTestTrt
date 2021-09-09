//#include "logger.h"
//#include "logging.h"


#include "utils.h"
#include <unistd.h>
// #include "scanfFile.h"
//#define  numOfAttribute 11   //���Եĸ�����
//using namespace nvcaffeparser1;
using namespace cv;
using namespace std;

static const int INPUT_H = 128; //输入高
static const int INPUT_W = 128;//输入宽
static const int CHANNELS = 3;//通道数
static const int OUTPUT_SIZE = 2;//输出
const char* INPUT_BLOB_NAME = "data";//输入层名
const char* OUTPUT_BLOB_NAME = "prob_1";//输出层名ffd

typedef  struct AttributeLabel
{
	int errorLabel = 0;
	int rightLabel = 0;
	int sumLabel = 0;
	int detectLabel = 0;
	float recall = 0;
	float accuracy = 0;
	int *pArray;

}Alabel;
void  getNumOfAttribute(const string &txtfileName, int *&ppAttrTagArray)
{
	//string  txtFileName = "H:/svn1/hz_object/project/hz_objectPedestrain/model/PedestrainGlobal.prototxt";
	ifstream  infile;
	infile.open(txtfileName, ios::in);
	vector<string> myStrArray;
	char myStr[1000];
	string outPutStr = "top: \"prob";  //ͨ���ж�����prob ���������ж��ж��ٸ�����
	int index = 0;
	while (infile.getline(myStr, 1000))
	{
		myStrArray.push_back(myStr);
		//cout << myStr << endl;	
	}
	infile.close();
	for (int i = 0; i < myStrArray.size(); i++)
	{
		string::size_type idx;
		idx = myStrArray[i].find(outPutStr);
		if (idx != string::npos)
		{
			index++;
		}
	}
	ppAttrTagArray = new int[index];

	/**ppAttrTagArray = pArray;*/

	//return  index;
}
vector<string>getLabelChinese()
{
	string txtName = "/home/cxl/tensorCaffe/AttributeTest/label_list_new1.txt";
	ifstream in(txtName, ios::in);
	string add_str;
	vector<string>strVec;
	if (!in)
	{
		cout << "���ļ�ʧ��" << endl;
		exit(0);
	}
	while (in)
	{
		getline(in, add_str);
		strVec.push_back(add_str);
	}
	strVec[0] = "";
	in.close();
	return strVec;
}
string getLabel1(const vector<string>& labelMap, int index) {
	if (index < 0 || index >= labelMap.size())
		return "*";

	return labelMap[index];
}

int main(int argc,char ** argv)
{
	// double allTimeBegin = static_cast<double>(cv::GetTickCount());
	cudaSetDevice(1);
	double allTimeBegin=static_cast<double>(getTickCount());
	int rightLabel = 0;
	int errorLabel = 0;
	int sumRightlabel = 0;
	// int AttributeNumber = 1;  //�������
	float sumTime = 0;
	bool saveStat = false;
	// char *saveMode ="-d";
	int count13 = 0;
	// char *attributeNumber="0";
	//int numOfAttribute = 11; //numOfAttributes
	//  char *modelFile = "/home/cxl/tensorCaffe/AttributeTest/model/pedestrainGlobal/PedestrainGlobal_babyCar_coatTemodel";
	//  char *deployFile = "/home/cxl/tensorCaffe/AttributeTest/model/pedestrainGlobal/PedestrainGlobal.prototxt";
	// argv[1]=deployFile;
	// argv[2]=modelFile;
	// strcpy(argv[3],saveMode);
	std::vector<std::string> outPutNameArray;  //����������
	//int attrTagArray[11];  //ÿ�����Ե�������С
	int *attrTagArray = nullptr;
	getNumOfAttribute(string(argv[1]), attrTagArray);
	auto numOfAttribute=getNameAndNumber(string(argv[1]), outPutNameArray, attrTagArray);   //���ÿ�����Ե���������ÿ���������������� ��prob_1,prob_2�ȵ�;
	// char *filePath = "/home/data/cxl/testData/pedestrain/0";   //����ȫ�������ļ���
	// argv[4]=filePath;
	// char *filePath = "/home/cxl/tensorCaffe/AttributeTest/testData";
	// char *trtSavePath = "/home/cxl/tensorCaffe/AttributeTest/build/pedestrain.engine";
	// argv[5]=trtSavePath;
	float mean_data[] = { 102.61518568033352, 101.82123008091003, 107.42755172448233 }; //pedestrain_global
	int AttributeNumber=atoi(argv[6]);
	//float mean_data[] = { 110.54368197 ,107.80291569 ,107.11519277 }; //vehicleSpecial mean
	//float mean_data[] = { 79.01901062, 78.72796895, 80.79338091 };//nostd_vehicle
	TrtClassificer *pMuclassifier=new TrtClassificer(INPUT_H, INPUT_W, CHANNELS, INPUT_BLOB_NAME, "prob", attrTagArray[0], numOfAttribute,attrTagArray,outPutNameArray);
	
	vector<string>fileList;
	vector<string> fileType={"jpg"};
	// vector<int>label;
    readFileList(argv[4],fileList,fileType);
	// readDir(filePath, fileList, label);
	 if (strcmp(argv[3],"-s")==0)
	 {
		 pMuclassifier->CaffeToGIEModel(argv[1], argv[2], outPutNameArray, 1, argv[5]); 
	 }
	 else if (strcmp(argv[3],"-d")==0)
	 {
		 Alabel  *pAlabel = new Alabel[attrTagArray[AttributeNumber]];
		 for (int i = 0; i < attrTagArray[AttributeNumber]; i++)
		 {
			 (pAlabel + i)->pArray = new int[attrTagArray[AttributeNumber]];
			 for (int j = 0; j < attrTagArray[AttributeNumber]; j++)
			 {
				 pAlabel[i].pArray[j] = 0;
			 }
		 }
		 pMuclassifier->readTrtModel(argv[5]);
		
		 int count = 0;
		//  float *feature = new float[attrTagArray[0]];
		 float *data = new float[INPUT_H*INPUT_W*CHANNELS];
	
		 for (auto &file : fileList)
		 {
			 imageProcess((char *)file.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);//Ԥ���� resize��ȥ��ֵ
			 float **prob3 = new float *[numOfAttribute];
			 for (int i = 0; i < numOfAttribute; i++)
			 {
				 prob3[i] = new float[attrTagArray[i]];
			 }
			//  clock_t starts1 = clock();
			 double starts1 = static_cast<double>(getTickCount());
			 pMuclassifier->doInferenceMultiOutPut(data, prob3, 1);  //forward ǰ�����
			//  clock_t ends1 = clock();
			 double ends1 = static_cast<double>(getTickCount());
			 sumTime += (ends1 - starts1)/getTickFrequency()*1000;
			 int max = findMax(prob3[AttributeNumber], attrTagArray[AttributeNumber]); //ʶ��ɵı�ǩ
			
			 int realLabel = getLabel(file); 
			 //��ʵ��ǩ

			 pAlabel[max].detectLabel++;
			 pAlabel[realLabel].sumLabel++;
			 //pAlabel[realLabel].errorRecognition[max]++;
			 pAlabel[realLabel].pArray[max]++;


			 if (max == realLabel)
			 {
				 cout << count++ << " " << file << " " << (ends1 - starts1)/(getTickFrequency())*1000 <<"  Right"<<endl;
				 /*	 if (max == 13)
					  {
						  count13++;
						  printf("count13=%d,filename=%s\n", count13, file.c_str());
					  }*/
				
				 rightLabel++;
				 pAlabel[max].rightLabel++;
			 }
			 else
			 {
				 cout << count++ << " " << file << " " << ends1 - starts1 << "     Error" << endl;
				 errorLabel++;
				 pAlabel[realLabel].errorLabel++;
				 
			 }
			 for (int i = 0; i < numOfAttribute; i++)
			 {
				 delete [] prob3[i];
			 }
			 delete[] prob3;
			
		 }
		 printf("\n\n%10s %10s %10s %10s %10s %10s\n", "Label", "sumLabels", "rightLabels", "Recall", "detect", "Accuracy");
		 for (int i = 0; i < attrTagArray[AttributeNumber]; i++)
		 {
			 pAlabel[i].recall = 1.0*pAlabel[i].rightLabel / pAlabel[i].sumLabel;
			 pAlabel[i].accuracy = 1.0*pAlabel[i].rightLabel / pAlabel[i].detectLabel;
			 printf("%10d %10d %10d %10.4f %10d %10.4f\n", i, pAlabel[i].sumLabel, pAlabel[i].rightLabel, pAlabel[i].recall, pAlabel[i].detectLabel, pAlabel[i].accuracy);
		 }
		 printf("%10s %10d %10d %10.4f\n\n", "ALL", fileList.size(), rightLabel, 1.0*rightLabel / fileList.size());
		 cout << "forward All Times: " << sumTime << "ms  Per pic: " << 1.0*sumTime / fileList.size() <<"ms"<< endl;
		 printf("\nError details:\n");
		 for (int i = 0; i < attrTagArray[AttributeNumber]; i++)
		 {
			 printf("\n%d recognizedAs :\n", i);
			 for (int j = 0; j < attrTagArray[AttributeNumber]; j++)
			 {
				 if(j!=i)
				 printf("%d : %d\t",j, pAlabel[i].pArray[j]);
			 }
			 printf("\n");
			 
		 }

		//  delete [] feature;
		 delete [] data;
		 for (int i = 0; i < attrTagArray[AttributeNumber]; i++)
		 {
			 delete[] pAlabel[i].pArray;
			 //(pAlabel + i)->pArray = new int[attrTagArray[AttributeNumber]];
		 }
		 delete[] pAlabel;
	 }
	 else if(strcmp(argv[3],"-c")==0)    //图片分类
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
	//float mean_data[] = { 99.95327338, 96.27925874, 86.54154894 }; //均值
	//float mean_data[] = { 97.59758647 , 99.04790283, 104.8204798 };
			float *pdata = data;
			setMean(im, mean_data, pdata, CHANNELS, INPUT_H, INPUT_W);

              float **prob3 = new float *[numOfAttribute];
			 for (int i = 0; i < numOfAttribute; i++)
			 {
				 prob3[i] = new float[attrTagArray[i]];
			 }
			  double starts1 = static_cast<double>(getTickCount());
			 pMuclassifier->doInferenceMultiOutPut(data, prob3, 1);  //forward ǰ�����
			//  clock_t ends1 = clock();
			 double ends1 = static_cast<double>(getTickCount());
			 sumTime += (ends1 - starts1)/getTickFrequency()*1000;
			 int max = findMax(prob3[AttributeNumber], attrTagArray[AttributeNumber]);
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

			  for (int i = 0; i < numOfAttribute; i++)
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
	 delete[] attrTagArray;
	 double allTimeEnd = static_cast<double>(getTickCount());;
	 double timeGap = allTimeEnd - allTimeBegin;
	 double fruency = getTickFrequency();
	 double useTime = (allTimeEnd - allTimeBegin) / getTickFrequency();
	 cout<<endl;
	 cout << "Total program time:" << useTime << " s"<<endl;
	//system("pause");
	return 0;
}


int main_plate(int argc,char ** argv)
{
	cudaSetDevice(0);
	double allTimeBegin=static_cast<double>(getTickCount());
	int rightLabel = 0;
	int errorLabel = 0;
	int sumRightlabel = 0;
	// int AttributeNumber = 1;  //�������
	float sumTime = 0;
	bool saveStat = false;
	char *saveMode ="-d";
	int count13 = 0;
	int tabelsnum = 3622;
	int time_step = 21;
	int alphabet_size = tabelsnum;
	// char *attributeNumber="0";
	//int numOfAttribute = 11; //numOfAttributes
	//  char *modelFile = "/home/cxl/tensorCaffe/AttributeTest/carPlate/squeeze-48x168_iter_38000.caffemodel";
	//  char *deployFile = "/home/cxl/tensorCaffe/AttributeTest/carPlate/deploy_resnet_noblstm_deep_384_conv_78_chinese3.prototxt";
	// argv[1]=deployFile;
	// argv[2]=modelFile;
	strcpy(argv[3],saveMode);
	std::vector<std::string> outPutNameArray{"fc1x2"};  //����������
	//int attrTagArray[11];  //ÿ�����Ե�������С
	int numOfAttribute = 1;
	int *attrTagArray = new int[numOfAttribute];
	attrTagArray[0]=tabelsnum*time_step;
	// getNumOfAttribute(string(argv[1]), attrTagArray);
	// auto numOfAttribute=getNameAndNumber(string(argv[1]), outPutNameArray, attrTagArray);   //���ÿ�����Ե���������ÿ���������������� ��prob_1,prob_2�ȵ�;
	// char *filePath = "/home/cxl/data/huolala";   //����ȫ�������ļ���
	// argv[4]=filePath;
	// char *filePath = "/home/cxl/tensorCaffe/AttributeTest/testData";
	// char *trtSavePath = "/home/cxl/tensorCaffe/AttributeTest/build/carPlate.engine";
	// argv[5]=trtSavePath;
	// float mean_data[] = { 0, 0, 0 }; //pedestrain_global
	int AttributeNumber=atoi(argv[6]);
	//float mean_data[] = { 110.54368197 ,107.80291569 ,107.11519277 }; //vehicleSpecial mean
	float mean_data[] = { 79.01901062, 78.72796895, 80.79338091 };//nostd_vehicle
	TrtClassificer *pMuclassifier=new TrtClassificer(INPUT_H, INPUT_W, CHANNELS, INPUT_BLOB_NAME, "prob", attrTagArray[0], numOfAttribute,attrTagArray,outPutNameArray);
	
	vector<string>fileList;
	vector<string> fileType={"jpg"};
	// vector<int>label;
    readFileList(argv[4],fileList,fileType);
	// readDir(filePath, fileList, label);
	 if (strcmp(argv[3],"-s")==0)
	 {
		 pMuclassifier->CaffeToGIEModel(argv[1], argv[2], outPutNameArray, 1, argv[5]); 
	 }
	 else {
		// int time_step = 21;
		// int alphabet_size = tabelsnum;
		float * prob2 = new float[attrTagArray[0]];
		pMuclassifier->readTrtModel(argv[5]);

		int count = 0;
		float *feature = new float[attrTagArray[0]];
		float *data = new float[INPUT_H*INPUT_W*CHANNELS];

		for (auto &file : fileList)
		{
			imageProcess((char *)file.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);//Ԥ���� resize��ȥ��ֵ
			float **prob3 = new float *[numOfAttribute];
			for (int i = 0; i < numOfAttribute; i++)
			{
				prob3[i] = new float[attrTagArray[i]];
			}
			auto starts1 = cv::getTickCount();
			pMuclassifier->doInferenceMultiOutPut(data, prob3, 1);  //forward ǰ�����
			auto ends1 = cv::getTickCount();
			sumTime += (ends1 - starts1);
			cout << (ends1 - starts1)/cv::getTickFrequency()*1000 << endl;
			int k = 0;
			for (int i = 0; i < time_step; i++)
			{
				for (int j = 0; j < alphabet_size; j++)
					prob2[k++] = prob3[0][i + j * time_step];
			}
			int blank_label = 0;
			int prev_label = blank_label;
			string result;
			vector<string> labelMap = getLabelChinese();

			for (int i = 0; i < time_step; ++i)
			{
				float* lin = prob2 + i * tabelsnum;
				int predict_label = std::max_element(lin, lin + tabelsnum) - lin;
				//cout << "predict_label: " << predict_label << endl;

				if (predict_label != blank_label && predict_label != prev_label)
				{
					/*float pro = 1 / (1 + exp(-(lin[predict_label] - 10.2) / 4));
					pro_temp.push_back(pro);
					if (pro < min)
					{
					min = pro;
					}*/
					result = result + getLabel1(labelMap, predict_label);
				}

				prev_label = predict_label;
			}
			cout << result << endl;
			for (int i = 0; i < numOfAttribute; i++)
			{
				delete[] prob3[i];
			}
			delete[] prob3;

		}
		
		delete[] feature;
		delete[] data;
		delete[] prob2;
	 }
	 cout<<endl;
	 delete pMuclassifier;
	 delete[] attrTagArray;
	 double allTimeEnd = static_cast<double>(getTickCount());;
	 double timeGap = allTimeEnd - allTimeBegin;
	 double fruency = getTickFrequency();
	 double useTime = (allTimeEnd - allTimeBegin) / getTickFrequency();
	 cout<<endl;
	 cout << "Total program time:" << useTime << " s"<<endl;
	//system("pause");
	return 0;
}