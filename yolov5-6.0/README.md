# 一、数据集
VOCdevkit/VOC2007下存放三个文件：
JPEGImages：原图
Annotations：用labelimg标注的xml文件
predefined_classes.txt，存放标签

1. 标注数据集
`pip install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple`，安装labelimg
2. cd 到VOC2007文件夹下，`labelimg JPEGImages predefined_classes.txt`开始标注
3. 标注格式选PascalVOC，保存文件夹设置为Annotations。
4. 运行`xml_to_txt.py`，将xml格式转成yolo需要的txt格式，并自动划分好训练和验证集。

# 二、训练
1. 下载预训练权重，yolo源代码地址：`https://github.com/ultralytics/yolov5/tree/v5.0`，
有`yolov5s.pt`等多种权重可选择。一般性能不太好的机器用`yolov5s.pt`，服务器上一般用`yolov5l.pt`
2. 修改数据配置文件。
`data/voc.yaml`文件复制一份，改成`ballon.yaml`,修改好训练、验证、测试文件的路径，类别数，类别名即可，其他注释掉。
`models/yolo5s.yaml`文件复制一份，改成`yolo5s_ballon.yaml`,只修改别数即可。
3. 修改train.py中的参数设置
```
parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')
parser.add_argument('--cfg', type=str, default='models/yolov5s_ballon.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default=ROOT / 'data/ballon.yaml', help='dataset.yaml path')
# 显存不够就适当调小一点
parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
# 进程数，先设为0，没问题再往上加。
parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
```

# 四、训练
```
python train.py --weights weights/yolov5s.pt --cfg models/yolov5s_ballon.yaml --data data/ballon.yaml
# 接上次训练
python train.py --weights weights/yolov5s.pt --cfg models/yolov5s_ballon.yaml --data data/ballon.yaml --resume runs/train/exp3/weights/last.pt 
```
# 五、检测
```
python detect.py --weights yolov5s.pt --source data/images --project runs/detect
```
# 六、模型评估
```
tensorboard --logdir=runs/train/exp # 打开tensorboard查看模型训练效果
```

create_dataloader中包含了多种数据增强方式

