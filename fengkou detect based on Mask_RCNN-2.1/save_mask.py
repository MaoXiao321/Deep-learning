import os
import sys
import json
import datetime
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage.draw
import yaml
from PIL import Image

from config import Config
import utils
import model as modellib
from queue import Queue

# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


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

    def draw_mask(self, num_obj, mask, image, image_id):
        """重写draw_mask"""
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        # 把mask存起来，下次读入时直接读
        tmp_path = os.path.join(info["path"].split("\\")[0], info["path"].split("\\")[1], 'rwmask',
                                info["path"].split("/")[-1].split(".")[0])
        np.savez_compressed(tmp_path, mask)
        return mask

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
        assert subset in ["train", "val"]
        dataset_root_path = os.path.join(dataset_root_path, subset)
        img_floder = os.path.join(dataset_root_path, 'pic')
        mask_floder = os.path.join(dataset_root_path, 'cv2_mask')
        # mask_floder = os.path.join(dataset_root_path, 'rwmask')
        imglist = os.listdir(img_floder)
        count = len(imglist)
        # yaml_floder = os.path.join(dataset_dir, 'labelme_json')
        for i in range(count):
            # 获取图片宽和高
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            # mask_path = mask_floder + "/" + filestr + ".npz"
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
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        # 上面四句注释掉，换成下面这句
        # mask = np.load(info['mask_path'])['arr_0']
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


if __name__ == '__main__':
    import argparse

    # 设置参数，推理时只要改command,weights和image/video
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fengkou.')
    parser.add_argument("--command", metavar="<command>", help="'train' or 'splash'", default="train")
    parser.add_argument('--dataset', required=False, metavar="/path/to/balloon/dataset/"
                        , help='Directory of the Balloon dataset', default="fengkou")
    parser.add_argument('--weights', required=False, metavar="/path/to/weights.h5"
                        , help="Path to weights .h5 file or 'coco'", default="coco")
    parser.add_argument('--logs', required=False, metavar="/path/to/logs/", help='Logs and checkpoints directory'
                        , default=DEFAULT_LOGS_DIR)
    parser.add_argument('--image', required=False, metavar="path or URL to image",
                        help='Image to apply the color splash effect on'
                        , default="fengkou/val/pic"
                        )
    parser.add_argument('--video', required=False, metavar="path or URL to video",
                        help='Video to apply the color splash effect on'
                        )
    args = parser.parse_args()

    # # Configurations
    # if args.command == "train":
    #     config = ShapesConfig()
    # else:
    #     class InferenceConfig(ShapesConfig):
    #         # Set batch size to 1 since we'll be running inference on
    #         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #         GPU_COUNT = 1
    #         IMAGES_PER_GPU = 1
    #
    #
    #     config = InferenceConfig()
    # config.display()
    #
    # # Create model，创建好网络结构
    # if args.command == "train":
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)
    # # 设置权重文件的路径
    # if args.weights.lower() == "coco":
    #     weights_path = COCO_WEIGHTS_PATH
    #     # Download weights file
    #     if not os.path.exists(weights_path):
    #         utils.download_trained_weights(weights_path)
    # elif args.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = model.find_last()[1]
    # elif args.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = model.get_imagenet_weights()
    # else:
    #     weights_path = args.weights
    #
    # # 加载权重文件
    # print("Loading weights ", weights_path)
    # if args.weights.lower() == "coco":
    #     # Exclude the last layers because they require a matching
    #     # number of classes
    #     model.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])
    # else:
    #     model.load_weights(weights_path, by_name=True)
    #
    # # Train or evaluate
    # if args.command == "train":
    #     train(model)
    # elif args.command == "splash":
    #     for name in os.listdir(args.image):
    #         image_path = os.path.join(args.image, name)
    #         detect_and_color_splash(model, image_path=image_path,
    #                                 video_path=args.video)

    dataset_train = DrugDataset()
    dataset_train.load_shapes(args.dataset, 'val')
    dataset_train.prepare()
    print("dataset_train-->", dataset_train.image_ids)

    queue = Queue()
    thread_num = 30
    [queue.put(id) for id in dataset_train.image_ids]
    print("kaishi ")
    pthread = Pool(thread_num)

    while True:
        image_id = queue.get()
        pthread.apply_async(dataset_train.load_mask, args=(image_id,))
        if queue.empty():
            break
    pthread.close()
    pthread.join()
