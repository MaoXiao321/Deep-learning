import os


def mkdir(subset):
    ROOT_DIR = os.getcwd()
    path = os.path.join(ROOT_DIR, subset)
    if subset in ['test', 'detect']:
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        if not os.path.exists(path):
            for name in ['pic', 'json', 'labelme_json', 'orig_mask', 'cv2_mask', 'rwmask']:
                tmp = os.path.join(path, name)
                os.makedirs(tmp)
    print('{}创建成功！'.format(path))


if __name__ == '__main__':
    # 这里是要创建新目录的地址
    for subset in ['train', 'val', 'test', 'detect']:
        mkdir(subset)
