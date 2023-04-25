
import torch
from torch import nn

from torch.autograd import Variable
from torch.nn import functional as F

class depthwise_activate(nn.Module):
    def __init__(self, ch_in):
        super(depthwise_activate, self).__init__()
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_in, kernel_size=1)

    def forward(self, x):
        x1 = self.depth_conv(x)
        x1 = self.point_conv(x)
        out = torch.maximum(x,x1)
        return out

def feature_fuse(feat_1, feat_2):
    """
        fuse two net's feature
    """
    att_map = torch.sigmoid(feat_1 * feat_2)
    y1 = feat_1 * att_map + feat_1
    y2 = feat_2 * att_map + feat_2
    out = torch.cat((y1, y2), dim=1)
    return out

##########################################################################

##########################################################################
class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
        
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.conv4 = conv(n_feat, n_feat, kernel_size, bias=bias)
    def forward(self, x,x_last, r_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + r_img
        x2 = torch.sigmoid(self.conv3(img))
        x_last = self.conv4(x_last)
        x1 = x1*x2
        x1 = x1+x+x_last
        return x1


####################################################################
## Unet
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            depthwise_activate(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            depthwise_activate(out_ch))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            depthwise_activate(out_ch)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_dense(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(conv_dense, self).__init__()

        n1 = 20
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = feature_fuse(e4, d5)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = feature_fuse(e3, d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = feature_fuse(e2, d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = feature_fuse(e1, d2)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out

################################################################
class conv_dense2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        pad_x = 1   # 若输出大小不变， pad_x = int(dilation * (kernel - 1) / 2)
        self.dense =  nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size = 3, padding = pad_x),
            SEBlock(output_dim,output_dim*6),
            # nn.ReflectionPad2d((1, 0, 1, 0)),
            # nn.AvgPool2d(2, stride=1),
            nn.BatchNorm2d(output_dim),
            depthwise_activate(output_dim))

    def forward(self,x,x_last,r_img):
        x1 = self.dense(x)

        return x1

class net2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        num_c1 = 6
        num_c2 = 48
        num_c3 = 48
        num_c4 = 48
        num_c5 = 48
        num_c6 = 48
        num_c7 = 48
        num_c8 = 48
        # num_c9 = 24

        self.conv1 = nn.Conv2d(3, num_c1, kernel_size = 3, padding = 1)
        self.dense1 = nn.ModuleList(
                    [conv_dense2(num_c1+3,num_c2)]+
                    [conv_dense2(num_c2,num_c2) for i in range(3)]
                    
        )
        self.dense2 = nn.ModuleList(
                    [conv_dense2(num_c2+num_c1+3,num_c3)]+
                    [conv_dense2(num_c3,num_c3) for i in range(3)]+
                    [SAM(num_c3, 1, False)]
        )
        self.dense3 = nn.ModuleList(
                    [conv_dense2(num_c3+num_c2+3,num_c4)]+
                    [conv_dense2(num_c4,num_c4) for i in range(3)]+
                    [SAM(num_c4, 1, False)]
        )
        self.dense4 = nn.ModuleList(
                    [conv_dense2(num_c4+num_c3+3,num_c5)]+
                    [conv_dense2(num_c5,num_c5) for i in range(3)]+
                    [SAM(num_c5, 1, False)]
        )
        self.dense5 = nn.ModuleList(
                    [conv_dense2(num_c5+num_c4+3,num_c6)]+
                    [conv_dense2(num_c6,num_c6) for i in range(3)]+
                    [SAM(num_c6, 1, False)]
        )
        self.dense6 = nn.ModuleList(
                    [conv_dense2(num_c6+num_c5+3,num_c7)]+
                    [conv_dense2(num_c7,num_c7) for i in range(3)]+
                    [SAM(num_c7, 1, False)]
        )
        self.dense7 = nn.ModuleList(
                    [conv_dense2(num_c7+num_c6+3,num_c8)]+
                    [conv_dense2(num_c8,num_c8) for i in range(3)]+
                    [SAM(num_c8, 1, False)]
        )
        # self.dense8 = nn.ModuleList(
        #             [conv_dense2(num_c8+num_c7+3,num_c9)]+
        #             [conv_dense2(num_c9,num_c9) for i in range(3)]+
        #             [SAM(num_c9, 1, False)]
        # )
        self.conv2 = nn.Conv2d(num_c8+3, 3, kernel_size = 3, padding = 1)
        self.BN = nn.BatchNorm2d(num_c8)
        
    def forward(self, x):
        x1 = F.relu( self.conv1(x))
        x2 = torch.cat((x1,x),1)
        for m in self.dense1:
            x2 = m(x2,x1,x)
        x3 = torch.cat((x2,x1,x),1)
        for m in self.dense2:
            x3 = m(x3,x2,x)
        x4 = torch.cat((x3,x2,x),1)
        for m in self.dense3:
            x4 = m(x4,x3,x)
        x5 = torch.cat((x4,x3,x),1)
        for m in self.dense4:
            x5 = m(x5,x4,x)
        x6 = torch.cat((x5,x4,x),1)
        for m in self.dense5:
            x6 = m(x6,x5,x)
        x7 = torch.cat((x6,x5,x),1)
        for m in self.dense6:
            x7 = m(x7,x6,x)
        x8 = torch.cat((x7,x6,x),1)
        for m in self.dense7:
            x8 = m(x8,x7,x)
        # x9 = torch.cat((x8,x7,x),1)
        # for m in self.dense8:
        #     x9 = m(x9,x8,x)
        x9 = torch.relu(self.BN(x8))
        # x10 = torch.cat((x9,x),1)
        # x10 = torch.relu(self.conv2(x10))
        
        return x9


class G_CGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        num_c1= 10
        num_c2= 40
        num_c3= 40
        num_c4= 40

        self.dense1 = conv_dense(3,num_c1)
        self.dense2 = conv_dense(num_c1,num_c1)
        
        self.dense3 =conv_dense(3,num_c1)
                    
        self.dense4 = conv_dense(num_c1,num_c1)
                    
        self.dense5 = conv_dense(3,num_c1)
                    
        self.dense6 = conv_dense(num_c1,num_c1)

        self.conv2 = nn.Conv2d(num_c1*3+3, num_c2, kernel_size = 3, padding = 1)
        self.BN1 = nn.BatchNorm2d(num_c2)
        self.activate1 = depthwise_activate(num_c2)
        self.conv3 = nn.Conv2d(num_c2+3, num_c3, kernel_size = 3, padding = 1)
        self.BN2= nn.BatchNorm2d(num_c3)
        self.activate2 = depthwise_activate(num_c3)
        self.conv4 = nn.Conv2d(num_c3+3, num_c4, kernel_size = 3, padding = 1)
        self.BN3= nn.BatchNorm2d(num_c4)
        self.activate3 = depthwise_activate(num_c4)
        self.conv5 = nn.Conv2d(num_c4+3, 3, kernel_size = 3, padding = 1)
        # self.net2 = net2()
        self.SAM1 = SAM(num_c1*1, 1, False)
        self.SAM2 = SAM(num_c1*1, 1, False)
        self.SAM3 = SAM(num_c1*1, 1, False)
        self.SAM4 = SAM(num_c1*3, 1, False)
        
        self.conv_edge = nn.Conv2d(num_c1*1, 3, kernel_size = 3, padding = 1)
        self.conv_ssim = nn.Conv2d(num_c1*1, 3, kernel_size = 3, padding = 1)
        self.conv_mse = nn.Conv2d(num_c1*1, 3, kernel_size = 3, padding = 1)

    def forward(self, x):
        x1 = self.dense1(x)
        x2 = self.dense2(x1)
        x3 = self.dense3(x)
        x4 = self.dense4(x3)
        x5 = self.dense5(x)
        x6 = self.dense6(x5)


        stage1_feat1 = x2
        stage1_feat2 = x4
        stage1_feat3 = x6
        stage1_sam1 = self.SAM1(stage1_feat1,stage1_feat1,x)
        stage1_sam2 = self.SAM2(stage1_feat2,stage1_feat2,x)
        stage1_sam3 = self.SAM3(stage1_feat3,stage1_feat3,x)
        mse = self.conv_edge(stage1_sam1)
        ssim = self.conv_ssim(stage1_sam2)
        edge = self.conv_ssim(stage1_sam3)
        # rain_feat = self.net2(x)
        # total fuse
        stage1_feat4 = torch.cat((stage1_sam1, stage1_sam2, stage1_sam3), 1)
        stage1_sam4 = self.SAM4(stage1_feat4,stage1_feat4,x)

        x9 = torch.cat((stage1_sam4, x), 1)
        x9 = self.BN1(self.conv2(x9))
        x9 = self.activate1(x9)

        x10 = torch.cat((x9, x), 1)
        x10 = self.BN2(self.conv3(x10))
        x10 = self.activate2(x10)

        x11 = torch.cat((x10, x), 1)
        x11 = self.BN3(self.conv4(x11))
        x11 = self.activate3(x11)

        x12 = torch.cat((x11, x), 1)
        final = torch.sigmoid(self.conv5(x12))

        return {'edge':edge, 'ssim':ssim, 'mse':mse, 'final':final}




if __name__ == '__main__':
    ts = torch.Tensor(8, 3,128,  128)
    vr = Variable(ts)
    G_net = G_CGAN()
    # G_net = depthwise_activate(3)
    # print(G_net)
    oups = G_net(vr)
    print(oups['final'].size())

    # import torchsummary
    # torchsummary.summary(G_CGAN().cuda(),(3,64,  64))
