from torch import nn
import torch
import torch.nn.functional as F
import torch as t
from read_data import ReadData, test_readdata
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import settings
import os
from model import G_CGAN
import cv2
from loss import SSIM
from utils import calculate_ssim,calculate_psnr


batch_size = 1
num_worker = 0

device = t.device('cuda' if t.cuda.is_available else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = settings.GPU_id 

class Session():
    def __init__(self):
        self.image_root = settings.test_image_root
        self.result_root = settings.test_result_root
        self.model_root = settings.model_root
        # 创建模型
        self.net = G_CGAN().cuda()

        self.img_num = 0
        self.ssim = SSIM().cuda()
    def load_model(self, name = 'latest'):
        model_path = os.path.join(self.model_root, name)
        try:
            obj = t.load(model_path)
            print('Load checkpoint %s' % model_path)
        except FileNotFoundError:
            
            print('No checkpoint %s!!' % model_path)
            return 
        self.net.load_state_dict(obj['G_net'])
        

    def save_img(self,  predict,original):
        for imgA, imgB in zip(predict, original):
            # 从tensor转回np
            imgA = np.array(imgA.cpu().data)  
            imgB =  np.array(imgB.cpu().data)
            # 拼接图片
            imgA = imgA.transpose(1,2,0) *255   # 原数据为0～1, 转回0～255
            imgB = imgB.transpose(1,2,0) *255   # 原数据为0～1, 转回0～255
            temp_img = np.hstack((imgA,imgB))  #未转换通道前是竖着拼接
            
            cv2.imwrite(self.result_root + str(self.img_num) + '.png', temp_img)
            print(str(self.img_num) + '.png')
            self.img_num = self.img_num +1
        return imgA, imgB
            
                
    @t.no_grad()   
    def test_model(self, model_name = 'latest'):
        #读取待测试数据
        test_data = test_readdata(self.image_root)
        dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=False, num_workers=num_worker)
        
        #载入模型参数
        self.load_model(model_name)
        self.net.eval()

        total_ssim = 0
        total_psnr = 0
        for i_batch,(imgA,imgB) in enumerate(dataloader):
            imgA = imgA.to(device) #使用cuda计算
            imgB = imgB.to(device) #使用cuda计算
            # r_img = imgA - imgB
            results = self.net(imgA)
            result, imgb = self.save_img( results['final'], imgB)
            
            img_ssim = calculate_ssim(result, imgb)
            img_psnr = calculate_psnr(result, imgb)
            print("ssim = ", img_ssim.item())
            total_ssim += img_ssim.item()
            print("psnr = ", img_psnr)
            total_psnr += img_psnr
        print("total_ssim = ", total_ssim/(i_batch+1))
        print("total_psnr = ", total_psnr/(i_batch+1))
if __name__ == '__main__':
    sess = Session()
    sess.test_model(settings.saved_model)#'best_model'    

# from torch import nn
# import torch
# import torch.nn.functional as F
# import torch as t
# from read_data import ReadData, test_readdata
# from torch.utils.data import DataLoader,Dataset
# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision
# import settings
# import os
# from model import G_CGAN
# import cv2
# from cal_ssim import SSIM



# batch_size = 1
# num_worker = 0

# device = t.device('cuda' if t.cuda.is_available else 'cpu')




# class Session():
#     def __init__(self):
#         self.image_root = settings.test_image_root
#         self.result_root = settings.test_result_root
#         self.model_root = settings.model_root
#         # 创建模型
#         self.net = G_CGAN().cuda()

#         self.img_num = 0
#         self.ssim = SSIM().cuda()
#     def load_model(self, name = 'latest'):
#         model_path = os.path.join(self.model_root, name)
#         try:
#             obj = t.load(model_path)
#             print('Load checkpoint %s' % model_path)
#         except FileNotFoundError:
            
#             print('No checkpoint %s!!' % model_path)
#             return 
#         self.net.load_state_dict(obj['G_net'])
        

#     def save_img(self, original, predict):
#         for imgA, imgB in zip(predict, original):
#             # 从tensor转回np
#             imgA = np.array(imgA.cpu().data)  
#             imgB =  np.array(imgB.cpu().data)
#             # 拼接图片
#             imgA = imgA.transpose(1,2,0) *255   # 原数据为0～1, 转回0～255
#             imgB = imgB.transpose(1,2,0) *255   # 原数据为0～1, 转回0～255
#             temp_img = np.hstack((imgA,imgB))  #未转换通道前是竖着拼接
            
#             cv2.imwrite(self.result_root + str(self.img_num) + '.png', temp_img)
#             self.img_num = self.img_num +1
            
                
#     @t.no_grad()   
#     def test_model(self, model_name = 'latest'):
#         #读取待测试数据
#         test_data = test_readdata(self.image_root)
#         dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=False, num_workers=num_worker)
        
#         #载入模型参数
#         self.load_model(model_name)
#         self.net.eval()

#         total_ssim = 0

#         for i_batch,(imgA,imgB) in enumerate(dataloader):
#             imgA = imgA.to(device) #使用cuda计算
#             imgB = imgB.to(device) #使用cuda计算
#             # r_img = imgA - imgB
#             results = self.net(imgA)
#             self.save_img(imgB, results)
            
#             img_ssim = self.ssim(results, imgB)
#             print("ssim = ", img_ssim.item())
#             total_ssim += img_ssim.item()
#         print("total_ssim = ", total_ssim/(i_batch+1))
# if __name__ == '__main__':
#     sess = Session()
#     sess.test_model(settings.saved_model)







