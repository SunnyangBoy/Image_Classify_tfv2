"""
    训练部分配置
"""

# 训练集图片文件夹路径
train_img = './data/sample/train_data/train_images'

# 训练集标注文件路径
train_txt = './data/sample/train_data/train.txt'

# checkpoint文件保存路径
h5_savePath = './saved_model/model_batch{batch}.h5'

# checkpoint保存间隔(step)
save_freq = 5000

# tensorboard文件保存路径
tensorboard_path = './logs'

# 预训练模型存放路径
premodel_path = './pretrain_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# 预训练模型是否可训练
trainable = True

# pb模型存放路径
pb_savepath = "./saved_models"

# 主干网络的选择模式：'Vgg16', 'ResNet50', None 表示采用浅层网络
base_model = None

# 训练迭代次数
epochs = 50

# 模型输出类型个数
num_classes = 4

# batch大小设置
batch_size = 8

# 输入图片的shape
input_shape = (600, 600, 3)

# 训练时图片resize大小
resize_shape = [600, 600]

# 训练学习率设置
base_learning_rate = 0.001

# 训练集与验证集的比例
train_and_val = 0.8


"""
    测试部分配置
"""

# 加载pb文件的路径
pb_loadpath = './saved_models/pb/'

# 测试集图片文件夹路径
test_img = './data/sample/test_data/test_images'

# 测试集图片文件夹路径
test_txt = './data/sample/test_data/test.txt'


"""
    预测部分配置
"""

# 待预测图片文件夹路径
infer_img = './data/sample/infer_data/infer_images'
