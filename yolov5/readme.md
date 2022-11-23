git clone https://github.com/ultralytics/yolov5
cd yolov5
# pip install -r requirements.txt
# mkdir dataset
# mkdir /home/data/xml
# mkdir /home/data/images
# mkdir /home/data/labels
# cp /home/data/1/*.xml /home/data/xml
# cp /home/data/1/*.jpg /home/data/images
# rm -rf dataset/*
# python /project/train/src_repo/yolov5/split_train_val.py --path /home/data/images --xml_path /home/data/xml  --txt_path /project/train/src_repo/yolov5/dataset
# python /project/train/src_repo/yolov5/voc_label.py

# wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
# python /project/train/src_repo/yolov5/train.py  --batch-size 8 --data ./data/brass.yaml --weights ./yolov5s.pt \
# --cfg ./models/yolov5s.yaml --project /project/train/models/ --epochs 2 --workers 0

python export.py --data ./data/brass.yaml --weights /project/train/models/exp/weights/best.pt --include onnx