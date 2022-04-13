"""
代码功能：掩码存储转换(将文件夹里的label.png从16转到8位)
"""

from PIL import Image
import numpy as np
import shutil
import os

from PIL import Image
import numpy as np
import shutil
import os

src_dir = r"./fengkou/val/orig_mask"
dest_dir = r"./fengkou/val/cv2_mask"

for label_name in os.listdir(src_dir):
    old_mask = os.path.join(src_dir, label_name)
    img = Image.open(old_mask)
    img = Image.fromarray(np.uint8(np.array(img)))
    new_mask = os.path.join(dest_dir, label_name)
    img.save(new_mask)


# s = "_json"
# # 8位的掩码文件
# file_path = r'C:\Users\DAIKIN\Desktop\xunlian'
# # 转成8位的掩码文件
# mask_save_path = "./Test"
# # 原图片位置
# image_save_path = "C://Users//DAIKIN//Desktop//xunlian//PNGImages//"
# filename = os.listdir(file_path)
# t = 0
# for name in filename:
#     # print("11111111111",name)
#     if s in name:
#         # print("name:",name.split('_')[0])
#         mask_file = os.listdir(file_path + "//" + name)
#         for maskname in mask_file:
#             if maskname.split('.')[0] == "label":
#                 old_mask_path = file_path + "//" + name + "//" + maskname
#                 # print("old_mask",old_mask_path)
#                 img = Image.open(old_mask_path)
#                 img = Image.fromarray(np.uint8(np.array(img)))
#                 if os.path.exists(mask_save_path):
#                     # print("11")
#                     img.save(mask_save_path + str(t) + "_mask.png")
#                 else:
#                     os.mkdir(mask_save_path)
#                     img.save(mask_save_path + str(t) + "_mask.png")
#             if maskname.split('.')[0] == "img":
#                 old_image = file_path + "//" + name + "//" + maskname
#                 if not os.path.exists(image_save_path):
#                     os.mkdir(image_save_path)
#                 shutil.copyfile(old_image, image_save_path + str(t) + ".jpg")
#         t = t + 1
