#! /bin/bash
/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/build/AttributePlate \
/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/model/laterReleaseModel/NoStdVehicle.prototxt \
/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/model/laterReleaseModel/NoStdVehicle.caffemodel \
-b \
/home/xiaolei/train_data/data/project/_testSets/NostdforAcc/handCartest \
/home/xiaolei/train_data/data/project/caffeAttributeTestTrt/engine/NoStdVehicle_batch8.engine \
4 \
/home/xiaolei/train_data/data/project/_testSets/clsFolder/test1/   \
prob_1,prob_2,prob_3,prob_4,prob_5,prob_6   \
11,11,4,3,2,5     
    
