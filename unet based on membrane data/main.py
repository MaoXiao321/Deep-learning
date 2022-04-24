from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 数据增强参数
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)


model = unet(pretrained_weights='unet_membrane.hdf5')
# 存储权重文件：unet_membrane.hdf5
model_checkpoint = ModelCheckpoint('unet_membrane1.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=300, epochs=4, callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)  # 30根据测试集数量设置
saveResult("data/membrane/test", results)
