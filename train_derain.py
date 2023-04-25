
from operator import le
from torch import nn

import torch
from read_data import ReadData
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import time
from model import G_CGAN

import settings
import cv2
import os
from torch.autograd import Variable

from cal_ssim import SSIM
from loss import EdgeLoss
import lpips
import logging
from tqdm import tqdm
# loss_fn_vgg = lpips.LPIPS(net='alex').to( torch.device('cuda'))

# def LpisLoss(out_train, target_train, device):

#     new_out_train = (torch.max(out_train)-out_train)/(torch.max(out_train)-torch.min(out_train))
#     new_target_train = (torch.max(target_train)-target_train)/(torch.max(target_train)-torch.min(target_train))
#     resize = transforms.Resize([256, 256])
#     new_target_train = resize(new_target_train)
#     new_out_train = resize(new_out_train)
#     lpips_num = 0
#     for ii in range(len(new_out_train)):
#         outtrain = new_out_train[ii].reshape((1,3,256,256))
#         targettrain = new_target_train[ii].reshape((1,3,256,256))
#         lpips_num += float(loss_fn_vgg(targettrain.to(device), outtrain.to(device)))
#         lpips_num = torch.tensor(lpips_num).to(device)

#         return lpips_num
logging.basicConfig(filename='./logdir/logger.log',format='%(asctime)s- %(message)s', level=logging.DEBUG)
os.environ['CUDA_VISIBLE_DEVICES'] = settings.GPU_id 
class Session:
    def __init__(self):
        self.image_root = settings.image_root
        self.label_root = settings.label_root
        self.model_root = settings.model_root
        self.logdir = settings.logdir
        self.saved_model = settings.saved_model

        self.batch_size = settings.batch_size 
        self.start_epoch = 0 
        self.max_epoch = settings.max_epoch
        self.lr = settings.lr  #学习率
        self.num_worker = settings.num_worker
        self.best_loss = 1

        self.val_interval = settings.val_interval  # 验证模型的间隔，单位epoch
        self.save_interval = settings.save_interval # 保存模型的间隔，单位为epoch

        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

         # 创建模型
        self.G_net = G_CGAN().cuda()


        # 定义损失函数和优化器
        self.G_loss = nn.MSELoss().cuda()
        self.ssim = SSIM().cuda()
        self.Edge_loss = EdgeLoss().cuda()

        self.G_optimizer = torch.optim.Adam(self.G_net.parameters(), lr=self.lr)

    


    def save_model(self, name):
        model_path = os.path.join(self.model_root, name)
        obj = {
            'G_net': self.G_net.state_dict(),
            'epoch': self.epoch,
            'G_opt': self.G_optimizer.state_dict(),

        }
        torch.save(obj, model_path)


    def load_model(self, name = 'latest'):
        model_path = os.path.join(self.model_root, name)
        try:
            obj = torch.load(model_path)
            logging.debug('Load checkpoint %s' % model_path)
        except FileNotFoundError:
            
            logging.debug('No checkpoint %s!!' % model_path)
            return 
        self.G_net.load_state_dict(obj['G_net'])

        self.G_optimizer.load_state_dict(obj['G_opt'])

        self.start_epoch = obj['epoch']


    def save_img(self, original, blackground, predict):
        for i in range(3):
            # 从tensor转回np
            imgA = np.array(original[i].cpu().data)  
            imgB = np.array(blackground[i].cpu().data)
            imgC =  np.array(predict[i].cpu().data)
            # 拼接图片
            temp_img = np.hstack((imgA, imgB, imgC))  #未转换通道前是竖着拼接
            temp_img = temp_img.transpose(1,2,0) *255   # 原数据为0～1, 转回0～255

            if 'img' in vars():  #判断img是否定义过
                img = np.hstack((img, temp_img)) #转换通道后是横着拼接
            else:
                img = temp_img
                
        cv2.imwrite(self.logdir + 'epoch_' + str(self.epoch) + '.png', img)

    #验证网络
    @torch.no_grad()
    def val_net(self, dataloader,net):
        net.eval()
        total_loss = 0
        total_ssim = 0
        for i_batch,(imgA,imgB) in enumerate(dataloader):
            imgA = imgA.to(self.device) #使用cuda计算
            imgB = imgB.to(self.device)
            results = net(imgA)
            loss = self.G_loss(results['final'],imgB) 
            ssim = self.ssim(results['final'], imgB)
            total_loss +=loss.item()
            total_ssim +=ssim.item()
        self.save_img(imgA, imgB, results['final'])
        logging.debug("*"*40)
        logging.debug("epoch [%d/%d]  val_loss =  %.4f val_ssim =  %.4f" % ( self.epoch, self.max_epoch,  total_loss/(i_batch+1), total_ssim/(i_batch+1)))
        logging.debug("*"*40)
        if total_loss/(i_batch+1) < self.best_loss:
            self.best_loss = total_loss/(i_batch+1)
            self.save_model('best_model')
            logging.debug("find a better model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return total_ssim/(i_batch+1)

    def trainval_model(self, model_name = 'latest'):
        #载入模型参数
        self.load_model(model_name)
        #读取数据
        train_data = ReadData(self.image_root,self.label_root, train = True)
        dataloader = DataLoader(train_data, batch_size=self.batch_size,shuffle=True, num_workers=self.num_worker)
        val_data = ReadData(self.image_root,self.label_root, train = False)
        val_loader = DataLoader(val_data, batch_size=self.batch_size,shuffle=True, num_workers=self.num_worker)
        
        #训练模型
        for self.epoch in range(self.start_epoch, self.max_epoch):
            G_total_loss = 0
            self.G_net.train()
            start_time = time.time()

            for i_batch, (imgA,imgB) in enumerate(tqdm(dataloader,0)):
                if imgA.shape[0] != self.batch_size:   #最后一个batch可能会剩下零头
                    continue
                imgA = imgA.to(self.device) #使用cuda计算
                imgB = imgB.to(self.device)
                # r_img = imgA - imgB
                # generator train
                
                results = self.G_net(imgA)
                edge_loss = self.Edge_loss(results['edge'], imgB)
                mse_loss = self.G_loss(results['mse'], imgB)
                ssim_loss = - self.ssim(results['ssim'], imgB)
                final_loss = - self.ssim(results['final'], imgB)
                
                G_loss = edge_loss + mse_loss + ssim_loss + final_loss
                
                self.G_optimizer.zero_grad()
                G_loss.backward()

                self.G_optimizer.step()
                
                G_total_loss +=G_loss.item()
                
            
            # 打印损失    
            end_time = time.time()
            logging.debug("epoch [%d/%d] i_batch %d G_loss =  %.4f  time = %.2f s" 
                    % ( self.epoch, self.max_epoch, i_batch, G_total_loss/i_batch, (end_time - start_time)))
            

            if self.epoch % self.val_interval == 0:
                
                val_loss = self.val_net(val_loader,self.G_net) #验证模型
                

                self.save_model('latest')  # 定时覆盖保存
                
            if self.epoch % self.save_interval == 0:
                self.save_model('epoch_' + str(self.epoch)+'_ssim_'+str(val_loss))

if __name__ == '__main__':
    sess = Session()
    while True:
        sess.trainval_model()




