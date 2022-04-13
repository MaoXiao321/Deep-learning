import os
import shutil

path = "labelme_json"
files = os.listdir(path)
for file in files:
    pic_name = file[:-5]+'.png'
    from_dir = './labelme_json/'+file+'/label.png'
    to_dir = './cv2_mask/' + pic_name
    shutil.copy(from_dir, to_dir)

    print(from_dir, to_dir)