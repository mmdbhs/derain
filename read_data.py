from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms 
import torchvision
import settings
import cv2
from numpy.random import RandomState
image_root = settings.image_root
label_root = settings.label_root
 



class ReadData(Dataset):
    def __init__(self,image_root,label_root,train = True):
        self.rand_state = RandomState(66)
        self.image_root = image_root
        self.label_root = label_root
        self.train = train
        self.data_filename = os.listdir(image_root)
        self.patch_size = settings.patch_size
        if train:
            self.data_filename = self.data_filename[0:int(len(self.data_filename)*1)]
        else:
            self.data_filename = self.data_filename[-int(len(self.data_filename)*0.2):]
        # self.label_filename = os.listdir(label_root)
        
    def __len__(self):
        
        return len(self.data_filename)

    def __getitem__(self, index):
        image_index = self.data_filename[index]
        image_path = os.path.join(self.image_root,image_index)


        img = cv2.imread(image_path).astype(np.float32) / 255    #label和result都在其中

        if settings.aug_data:
            O, B = self.crop(img, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
        else:
            O, B = self.crop(img, aug=False)
            O, B = self.flip(O, B)
            O, B = O.copy(), B.copy()
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        

        return O,B

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        B = img_pair[r: r+p_h, c+w: c+p_w+w]
        O = img_pair[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class test_readdata(Dataset):
    def __init__(self,image_root):
        self.image_root = image_root
        self.label_root = label_root
        self.data_filename = os.listdir(image_root)
    
    def __len__(self):
        
        return len(self.data_filename)

    def __getitem__(self, index):
        image_index = self.data_filename[index]
        image_path = os.path.join(self.image_root,image_index)

        img = cv2.imread(image_path).astype(np.float32) / 255    #label和result都在其中
        h, w, c = img.shape
        imgO = img[0:int(h),0:int(w/2-1)]
        imgB = img[0:int(h),int(w/2):int(w-1)]
        # imgO = cv2.resize(imgO, (512,512), cv2.INTER_AREA)
        # imgB = cv2.resize(imgB, (512,512), cv2.INTER_AREA)
        imgO = np.transpose(imgO, (2, 0, 1))
        imgB = np.transpose(imgB, (2, 0, 1))
        return imgO,imgB

def main():
    pass

if __name__ == '__main__':
    
    if False:
        train_data = ReadData(image_root,label_root, train = True)
        dataloader = DataLoader(train_data, batch_size=4,shuffle=True)
        for i_batch,data_batch in enumerate(dataloader): 
            print(i_batch)
            print(data_batch[0].size())
            plt.subplot(1,2,1)
            plt.imshow(np.transpose(torchvision.utils.make_grid(data_batch[0][0]),(1,2,0)))  #将tensor数据转化为图片输出
            plt.subplot(1,2,2)
            plt.imshow(np.transpose(torchvision.utils.make_grid(data_batch[1][0]),(1,2,0)))
            plt.show()
            break
        print(train_data.__len__())
    else:
        train_data = ReadData(image_root,label_root, train = True)
        dataloader = DataLoader(train_data, batch_size=1,shuffle=True)
        val_data = ReadData(image_root,label_root, train = False)
        val_loader = DataLoader(val_data, batch_size=1,shuffle=True)
        for i_batch,data_batch in enumerate(val_loader): 

            img = data_batch[0][0].data
            img= np.array(img)


            img2 = data_batch[1][0].data
            img2 = np.array(img2)

            
            img = np.hstack((img, img2))
            img = img.transpose(1,2,0)
            cv2.imshow('jjj.png',img)
            cv2.waitKey(0)
            break
    
    
