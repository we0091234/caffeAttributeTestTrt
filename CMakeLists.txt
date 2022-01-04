cmake_minimum_required(VERSION 3.5)
set(OpenCV_DIR "/home/cxl/myData/softWare/opencv-4.5.1/build")
project(caffeAttribute)
find_package(OpenCV REQUIRED)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.2/include/)
link_directories(/usr/local/cuda-10.2/lib64/)
# cudnn
include_directories(/home/cxl/myData/tensorRT/cuda/include)
link_directories(/home/cxl/myData/tensorRT/cuda/lib64)
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/home/cxl/myData/tensorRT/TensorRT-7.2.3.4/include)
link_directories(/home/cxl/myData/tensorRT/TensorRT-7.2.3.4/lib/)

# set(OpenCV_DIR usr/local/share/opencv4)
# find_package(OpenCV  REQUIRED)
# include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} /usr/local/include/opencv4)
# find_package("/usr/local/include/opencv4/opencv2")
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -mavx2 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#cuda_add_library(leaky ${PROJECT_SOURCE_DIR}/leaky.cu)
# cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
# target_link_libraries(yololayer nvinfer cudart ${OpenCV_LIBS})
# target_link_libraries(yololayer nvinfer cudart "/usr/local/lib/opencv4.1/")
# aux_source_directory(${PROJECT_SOURCE_DIR}/include SOURCEFILE)
add_executable(caffeAttribute myCaffeInfer_ori1.cpp  calibrator.cpp TrtClassificer.cpp utils.cpp)
# add_executable(Attribute ${PROJECT_SOURCE_DIR}/logger.cpp ${PROJECT_SOURCE_DIR}/myTensorRT.cpp ${PROJECT_SOURCE_DIR}/logger.h ${PROJECT_SOURCE_DIR}/common.h   ${PROJECT_SOURCE_DIR}/argsParser.h ${PROJECT_SOURCE_DIR}/buffers.h ${PROJECT_SOURCE_DIR}/half.h  ${PROJECT_SOURCE_DIR}/scanfFile.cpp)
target_link_libraries(caffeAttribute nvinfer)
target_link_libraries(caffeAttribute cudart)
target_link_libraries(caffeAttribute nvcaffe_parser)
# target_link_libraries(Attribute yololayer)
target_link_libraries(caffeAttribute ${OpenCV_LIBS})
# target_link_libraries(Attribute "/usr/local/lib/opencv4.1/")

add_definitions(-O0 -pthread)

