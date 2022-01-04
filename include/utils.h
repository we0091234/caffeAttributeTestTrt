#ifndef UTILS_H_
#define UTILS_H_
// #include "TrtClassificer.h"
#include <immintrin.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <string.h>
#include <dirent.h>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
#ifdef use_aligned
#define my_mm256_load_ps _mm256_load_ps
#define my_mm256_store_ps _mm256_store_ps
#define my_mm256_load_si256 _mm256_load_si256
#define my_mm256_store_si256 _mm256_store_si256
#else
#define my_mm256_load_ps _mm256_loadu_ps
#define my_mm256_store_ps _mm256_storeu_ps
#define my_mm256_load_si256 _mm256_loadu_si256
#define my_mm256_store_si256 _mm256_storeu_si256
#endif
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
std::string getHouZhui(std::string fileName);
int readFileList(char *basePath,std::vector<std::string> &fileList,std::vector<std::string> fileType);
int findMax(float *prob, int OUTPUT_SIZE);
void setMean(cv::Mat & im, float *mean_data, float *&pdata, int CHANNELS, int INPUT_H, int INPUT_W);
void imageProcess(char *picName, float *data,float *mean_data, int CHANNELS, int INPUT_H, int INPUT_W);
float getSimilarity(std::vector<float> &lhs, std::vector<float>& rhs,int n);
float getSimilarity1(float *lhs, float * rhs, int n);
void normalizex(int n,  float * data);
int dot_short_SSE(int len, const short* v1, const short* v2);
bool judge(const pair<string, float> a, const pair<string, float> b);
string  getPersonId(string & fileName);
int getLabel(string &file);
std::vector<std::string> split(std::string str, std::string pattern);
void  getNumOfAttribute(const string &txtfileName, int *&ppAttrTagArray);
vector<string>getLabelChinese();
string getLabel1(const vector<string>& labelMap, int index);
int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);


#endif

