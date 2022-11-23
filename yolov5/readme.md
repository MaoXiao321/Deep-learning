## 代码准备
`git clone https://github.com/ultralytics/yolov5`

## 安装依赖
`cd yolov5
pip install -r requirements.txt`

## 数据准备
原始数据存储在/home/data/1/下面，包含.jpg和.xml。创建以下目录：
```
mkdir dataset
mkdir /home/data/xml
mkdir /home/data/images
mkdir /home/data/labels
cp /home/data/1/*.xml /home/data/xml
cp /home/data/1/*.jpg /home/data/images
rm -rf dataset/*
```
划分数据集：
```
python /project/train/src_repo/yolov5/split_train_val.py --path /home/data/images --xml_path /home/data/xml  --txt_path /project/train/src_repo/yolov5/dataset
```

将xml转成txt：
```
python /project/train/src_repo/yolov5/voc_label.py
```

## 开始训练
data中新建brass.yaml，参照其他的yml文件写。
```
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
python /project/train/src_repo/yolov5/train.py  --batch-size 8 --data ./data/brass.yaml --weights ./yolov5s.pt --cfg ./models/yolov5s.yaml --project /project/train/models/ --epochs 2 --workers 0
```
## 将pt转为onnx
```
python export.py --data ./data/brass.yaml --weights /project/train/models/exp/weights/best.pt --include onnx
```
