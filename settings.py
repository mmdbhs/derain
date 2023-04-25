

image_root = '../data/rain100H/train/'
label_root = './data/training/ground_truth/'
model_root = './saved_model/'
logdir = './logdir/'
saved_model = 'latest'#'best_model'  #载入训练好的模型使用或者继续训练

test_image_root = '../data/rain100H/test/'
test_result_root = './result_img/'

batch_size = 48
patch_size = 128
max_epoch = 50000
lr = 5e-4  #学习率
num_worker =4  

val_interval = 10  # 验证模型的间隔，单位epoch
save_interval = 50 # 保存模型的间隔，单位为epoch

aug_data = False  # 数据增强   

GPU_id = '0'
