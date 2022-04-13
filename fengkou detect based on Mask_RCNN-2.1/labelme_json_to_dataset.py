"""
代码功能：利用json文件生成mask掩码数据
运行时在run>> edit configuration 中输入parameters 为json 文件所在的路径，编译器为labelme的python环境
"""

import os

path = "./fengkou/val/json"
files = os.listdir(path)
json_file = os.listdir(path)

# os.system("activate labelme")

for file in json_file:
    os.system("labelme_json_to_dataset.exe %s" % (path + '/' + file))

# with open('labelme_json_to_dataset.txt', 'w') as f:
#     for file in files:
#         if file.endswith('.json'):
#             path_c = path + file.split('.')[0] + "_json"
#             if os.path.exists(path_c):
#                 continue
#             else:
#                 order = "labelme_json_to_dataset " + file
#                 # print(order)
#                 # os.system("cd D:/work/learn/maskrcnn/PennFudanPed/Test")
#                 os.system(order)
#                 f.write(order + "\n")
#
# import os
#
# path = 'G:/labeled/json'  # path为json文件存放的路径
#
# json_file = os.listdir(path)
#
# os.system("activate labelme")
#
# for file in json_file:
#     os.system("labelme_json_to_dataset.exe %s" % (path + '/' + file))
