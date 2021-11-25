#! /bin/bash
# /home/xiaolei/train_data/data/project/caffeAttributeTestTrt/build/AttributePlate \
# /home/xiaolei/train_data/data/project/caffeAttributeTestTrt/releaseModel/nostdVehicle/NoStdVehicle-20211112.prototxt \
# /home/xiaolei/train_data/data/project/caffeAttributeTestTrt/releaseModel/nostdVehicle/NoStdVehicle-20211112.caffemodel \
# -d \
# /home/xiaolei/train_data/data/datasets/NostdTestSets/val5 \
# /home/xiaolei/train_data/data/project/caffeAttributeTestTrt/releaseModel/nostdVehicle/NoStdVehicle.engine \
# 5 \
# /home/xiaolei/train_data/data/project/_testSets/clsFolder/test1/   \
# prob_1,prob_2,prob_3,prob_4,prob_5,prob_6  \
# 11,11,4,3,2,5


/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/build/caffeAttribute \
/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/model/yolov3/ObjectDetectStageP3x160x96.prototxt \
/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/model/yolov3/ObjectDetectStageP3x160x96.caffemodel \
-d \
/home/xiaolei/train_data/data/datasets/NostdTestSets/val5 \
/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/model/yolov3/NoStdVehicle.engine \
0 \
/home/xiaolei/train_data/data/project/_testSets/clsFolder/test1/   \
layer21-conv \
585
    
