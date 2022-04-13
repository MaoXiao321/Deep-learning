# 一、数据集
fengkou：风口数据，运行mkdir.py将fengkou下的文件夹创建好。
pic文件夹：放原始图片
json：labelme标注后生成的json文件
labelme_json：`labelme_json_to_dataset.py`将json文件转成的文件夹
orig_mask：labelme_json文件夹中的掩码数据
cv2_mask：转为8位的掩码数据
 rwmask：掩码存储文件。通过`save_mask.py`将mask先存储好，训练时可直接读取，加快速度。源码中mask用了三层循环，太慢了。

# 二、权重文件
下载mask_rcnn_coco.h5，基于coco数据集<br>
下载mask_rcnn_balloon.h5，基于balloon数据集

# 三、标注数据
## 1. 启动labelme环境
命令labelme（pip install labelme）
## 2. 标注
labelme中打开图片所在文件夹（fengkou\train,fengkou\val），对图片上物体进行标识（画多边形），
同一类别的两个物体需要分开标注（person1，person2...）
## 3. 标注后save形成json文件
例如FudanPed00005.png标注后生成FudanPed00005.json文件
## 4. 利用json文件生成mask掩码数据
json文件仅仅是包含标注的坐标信息和类别，需要转为掩码数据。运行`labelme_json_to_dataset.py`即可。
### 手动步骤：
4.1 cd /maskrcnn/PennFudanPed/PNGImages进入json文件所在文件夹
4.2 labelme_json_to_dataset FudanPed00005.json，生成FudanPed00005_json文件夹
4.3 文件夹包含的内容有：原图img；标签数据label; 类别标签label_names; 视图展示label_viz。
4.4 如果转换后的json文件夹没有yaml文件，可更改labelme安装文件（\Lib\site-packages\labelme\cli）下的json_to_dataset.py。
加上这些语句即可。或者改下源码，不用yaml文件，因为它存储的其实就是类别信息。
```
import yaml
logger.warning('info.yaml is being replaced by label_names.txt')
info = dict(label_names=label_names)
with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
    yaml.safe_dump(info, f, default_flow_style=False)
```
5. 掩码存储转换：运行`labelchange_16to8.py`即可

# 四、训练
```
python fengkou.py --command train --dataset fengkou --weights coco
python fengkou.py --command train --dataset fengkou --weights last # 接上次继续训练
```
# 五、检测
```
python fengkou.py --command splash --image fengkou/test --weights last
```
# 六、模型评估
```
python fengkou.py --command evaluate --dataset fengkou --weights last
```

maskrcnn源码地址：https://github.com/matterport/Mask_RCNN

