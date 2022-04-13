import os
import sys
import json
import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage.draw
import yaml
from PIL import Image

from config import Config
import utils
import model as modellib
import visualize

# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 改类别数 (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 80
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class DrugDataset(utils.Dataset):
    def get_obj_index(self, image):
        """得到该图中有多少个实例（物体）"""
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        """解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签"""
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
            return labels

    # def draw_mask(self, num_obj, mask, image, image_id):
    #     """重写draw_mask"""
    #     info = self.image_info[image_id]
    #     for index in range(num_obj):
    #         for i in range(info['width']):
    #             for j in range(info['height']):
    #                 at_pixel = image.getpixel((i, j))
    #                 if at_pixel == index + 1:
    #                     mask[j, i, index] = 1
    #     # # 把mask存起来，下次读入时直接读
    #     # tmp_path = os.path.join(info["path"].split("\\")[0], info["path"].split("\\")[1], 'rwmask',
    #     #                  info["path"].split("/")[-1].split(".")[0])
    #     # np.savez_compressed(tmp_path, mask)
    #     return mask

    def load_shapes(self, dataset_root_path, subset):
        """重新写load_shapes，里面包含自己的自己的类别
           count: number of images to generate.
    　　　　height, width: the size of the generated images.
        """
        # 改成自己数据的类别
        self.add_class("shapes", 1, "fengkou")
        # self.add_class("shapes", 2, "class1")
        # self.add_class("shapes", 3, "class2")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]

        dataset_root_path = os.path.join(dataset_root_path, subset)
        img_floder = os.path.join(dataset_root_path, 'pic')
        # mask_floder = os.path.join(dataset_root_path, 'cv2_mask')
        mask_floder = os.path.join(dataset_root_path, 'rwmask')
        imglist = os.listdir(img_floder)
        count = len(imglist)
        # yaml_floder = os.path.join(dataset_dir, 'labelme_json')
        for i in range(count):
            # 获取图片宽和高
            filestr = imglist[i].split(".")[0]
            # mask_path = mask_floder + "/" + filestr + ".png"
            mask_path = mask_floder + "/" + filestr + ".npz"
            yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/info.yaml"
            cv_img = cv2.imread(dataset_root_path + "/labelme_json/" + filestr + "_json/img.png")
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i], width=cv_img.shape[1],
                           height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        """重写load_mask,Generate instance masks for shapes of the given image ID."""
        global iter_num
        # print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        # img = Image.open(info['mask_path'])
        # num_obj = self.get_obj_index(img)
        # mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        # mask = self.draw_mask(num_obj, mask, img, image_id)
        # 上面四句注释掉，换成下面这两句
        mask = np.load(info['mask_path'], allow_pickle=True)
        mask = mask['arr_0']
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            # 改成自己数据的类别标签
            if labels[i].find("fengkou") != -1:
                labels_form.append("fengkou")
            # if labels[i].find("class1") != -1:
            #     labels_form.append("class1")
            # if labels[i].find("class2") != -1:
            #     labels_form.append("class2")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def load_data():
    """Train the model."""
    dataset_train = DrugDataset()
    dataset_train.load_shapes(args.dataset, 'train')
    dataset_train.prepare()

    dataset_val = DrugDataset()
    dataset_val.load_shapes(args.dataset, 'val')
    dataset_val.prepare()
    return dataset_train, dataset_val


def train(model):
    """Train the model."""
    # train与val数据集准备
    dataset_train, dataset_val = load_data()
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    # 自己数据集很小时用迁移学习思想，只训练heads层或者更少的层即可
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
　　all visualizations in the notebook. Provide a
　　central point to control graph sizes.
　　Change the default size attribute to control the size
　　of rendered images
　　"""
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None, splash=False):
    """输出检测后的图像"""
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects，返回rois,class_ids,scores,masks
        r = model.detect([image], verbose=1)[0]
        savepath = os.path.join(image_path.split('/')[0], 'detect', image_path.split('\\')[-1])
        if splash:
            # Color splash，返回结果图片上每个像素点的RGB值
            splash = color_splash(image, r['masks'])
            # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            skimage.io.imsave(savepath, splash)
        else:
            ax = get_ax(1)
            class_names = ['BG', 'fengkou']
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, savepath, scores=r['scores']
                                        , title="Predictions", ax=ax)

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    # print("Saved to ", file_name)


def compute_batch_ap(dataset):
    APs = []
    for image_id in dataset.image_ids:
        print(image_id)
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        if len(r['rois']) == 0:
            AP = 0
        else:
            # precisions和recalls是个列表，看倒数第二位就是真实的得分
            # overlaps是一个矩阵，行：预测，列：gt，值：IOU
            # AP：精度指标，平均AP值，AP值就是P-R曲线下方面积。这里是对所有类别求平均
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs


if __name__ == '__main__':
    import argparse
    # 设置参数，推理时只要改command,weights和image/video
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fengkou.')
    parser.add_argument("--command", metavar="<command>", help="'train' or 'splash' or 'evaluate'", default="evaluate")
    parser.add_argument('--dataset', required=False, metavar="/path/to/balloon/dataset/"
                        , help='Directory of the Balloon dataset', default="fengkou")
    parser.add_argument('--weights', required=False, metavar="/path/to/weights.h5"
                        , help="Path to weights .h5 file or 'coco'", default="last")
    parser.add_argument('--logs', required=False, metavar="/path/to/logs/", help='Logs and checkpoints directory'
                        , default=DEFAULT_LOGS_DIR)
    parser.add_argument('--image', required=False, metavar="path or URL to image",
                        help='Image to apply the color splash effect on'
                        , default="fengkou/test")
    parser.add_argument('--video', required=False, metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = ShapesConfig()
    else:
        class InferenceConfig(ShapesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model，创建好网络结构
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    # 设置权重文件的路径
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # 加载权重文件
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        for name in os.listdir(args.image):
            image_path = os.path.join(args.image, name)
            detect_and_color_splash(model, image_path=image_path)
    elif args.command == "evaluate":
        dataset_train, dataset_val = load_data()
        APs_train = compute_batch_ap(dataset_train)
        APs_val = compute_batch_ap(dataset_val)
        print("train: mAP @ IoU=0.5: ", np.mean(APs_train))  # 默认iou阈值是0.5
        print("val: mAP @ IoU=0.5: ", np.mean(APs_val))  # 默认iou阈值是0.5
