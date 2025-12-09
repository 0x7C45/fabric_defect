# fabric_defect
## 本项目数据来自
```
https://tianchi.aliyun.com/dataset/79336
```
### 将测试数据集存放在data_test目录下
### 将训练数据集存放在data_train目录下，格式见该目录readme.md

## 运行方式

先执行根目录下convert_data.py完成数据格式到yolo数据格式的转换，完成后能够在根目录下看见yolo_dataset文件夹
之后就可以执行train.py了，不出意外训练完成可以看见train目录下的训练结果
解决中文乱码的方式在这里
```
https://www.cnblogs.com/EDG-Clearlove7777777/articles/18658270
```
我的运行环境是miniconda25.9.1-3 yolov8n，通过
```
conda create --name yolov8
conda activate yolov8
```
创建并激活环境
按照yolov8官方文档完成环境配置，之后直接在目录下执行程序即可
