#    Copyright 2023 Zifeng, Lian
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn
import blocks as bl
import DSAblock as DSA
import NLAblock as NLA
import CAblock as CA
import SAblock as SA
import os

params = {'out_channels': 64,
          'kernel_h': 3,
          'kernel_w': 3,
          'kernel_c': 2,
          'stride_conv': 1,
          'pool': 2,
          'stride_pool': 2,
          'drop_out': 0.1}


class ISsegATUNet(nn.Module):
    def __init__(self, img_size, out_channels):
        super(ISsegATUNet, self).__init__()

        params['in_channels'] = 2
        self.encode1 = bl.Encoder(params)
        params['in_channels'] = 64
        self.encode2 = bl.Encoder(params)
        self.encode3 = bl.Encoder(params)
        self.encode4 = bl.Encoder(params)
        self.encode5 = bl.Encoder(params)
        self.bottleneck = bl.tripleConv(params)

        # 三个DSA
        self.attentionblock1 = DSA.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))
        self.attentionblock2 = DSA.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))
        self.attentionblock3 = DSA.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))

        # 四个NLA
        self.sa4 = NLA.NONLocalBlock2D(in_channels=64, inter_channels=64 // 4)
        self.sa3 = NLA.NONLocalBlock2D(in_channels=64, inter_channels=64 // 4)
        self.sa2 = NLA.NONLocalBlock2D(in_channels=64, inter_channels=64 // 4)
        self.sa1 = NLA.NONLocalBlock2D(in_channels=64, inter_channels=64 // 4)

        # 四个CA
        self.CA4 = CA.SplAtConv2d(64, 64, 3, padding=1)
        self.CA3 = CA.SplAtConv2d(64, 64, 3, padding=1)
        self.CA2 = CA.SplAtConv2d(64, 64, 3, padding=1)
        self.CA1 = CA.SplAtConv2d(64, 64, 3, padding=1)

        # 四个编码与解码的拼接
        self.up_contact4 = bl.UpCat(64, 64, is_deconv=True)
        self.up_contact3 = bl.UpCat(64, 64, is_deconv=True)
        self.up_contact2 = bl.UpCat(64, 64, is_deconv=True)
        self.up_contact1 = bl.UpCat(64, 64, is_deconv=True)

        # SA之前的四个不同尺度特征上采样
        self.dsv4 = bl.Up(64, 16, scale_size=(img_size, img_size))
        self.dsv3 = bl.Up(64, 16, scale_size=(img_size, img_size))
        self.dsv2 = bl.Up(64, 16, scale_size=(img_size, img_size))
        self.dsv1 = bl.Up(64, 16, scale_size=(img_size, img_size))

        # 一个尺度注意力
        self.scale_att = SA.scale_atten_convblock(64, 4)

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=1)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))  # number of labels

    def forward(self, input):
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)
        out5 = self.bottleneck.forward(e4)

        up4 = self.up_contact4(out4, out5)
        up4 = self.conv4(up4)
        g_conv4, at4 = self.sa4(up4)
        up4, attw4 = self.CA4(g_conv4)


        g_conv3, att3 = self.attentionblock3(out3, up4)
        up3 = self.up_contact3(g_conv3, up4)
        # up3 = self.up_contact3(out3, up4)
        up3 = self.conv3(up3)
        g_conv3, at3 = self.sa3(up3)
        up3, attw3 = self.CA3(g_conv3)

        g_conv2, att2 = self.attentionblock2(out2, up3)
        up2 = self.up_contact2(g_conv2, up3)
        # up2 = self.up_contact2(out2, up3)
        up2 = self.conv2(up2)
        g_conv2, at2 = self.sa2(up2)
        up2, attw2 = self.CA2(g_conv2)

        g_conv1, att1 = self.attentionblock1(out1, up2)
        up1 = self.up_contact1(g_conv1, up2)
        # up1 = self.up_contact1(out1, up2)
        up1 = self.conv1(up1)
        g_conv1, at1 = self.sa1(up1)
        up1, attw1 = self.CA1(g_conv1)

        # 将不同尺度特征进行上采样，上采样的尺度特征与输入相同
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        # 拼接四个上采样后的特征
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        # 进行尺度注意力运算
        outimage, a2, a1 = self.scale_att(dsv_cat)
        # 最后经过卷积获取输出结果
        out_label = self.conv6(outimage)
        # 返回最终结果
        return out_label

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    from torch.autograd import Variable
    img = Variable(torch.zeros(2, 2, 256, 256)).cuda()
    net = ISsegATUNet(256, 2).cuda()
    out = net(img)
    print(net)
    print(out.size())