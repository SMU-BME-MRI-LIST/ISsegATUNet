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


class tripleConv(nn.Module):
    def __init__(self, params):
        super(tripleConv, self).__init__()

        # 填充使卷积不改变图像大小
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['in_channels'] + params['out_channels'])
        conv2_out_size = int(params['in_channels'] + params['out_channels'] + params['out_channels'])

        self.conv1 = nn.Conv2d(in_channels=params['in_channels'], out_channels=params['out_channels'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])

        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['out_channels'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])

        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['out_channels'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])

        self.batchnorm1 = nn.BatchNorm2d(num_features=params['in_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)

        self.prelu = nn.PReLU()

        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        # DenseNet操作：BN+ReLU+conv+拼接，三次
        o1 = self.batchnorm1(input)
        o2 = self.prelu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.prelu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input, o3, o7), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.prelu(o9)
        out = self.conv3(o10)
        return out


class UpCat(nn.Module):
    def __init__(self, in_channel, out_channel, is_deconv=True):
        super(UpCat, self).__init__()

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

    def forward(self, input, down_output):
        output = self.up(down_output)

        # 检查维度是否满足拼接条件
        offset = input.size()[3] - output.size()[3]
        if offset == 1:
            addition = torch.rand((output.size()[0], output.size()[1], output.size()[2]), out=None).unsqueeze(3).cuda()
            output = torch.cat([output, addition], dim=3)
        elif offset > 1:
            addition = torch.rand((output.size()[0], output.size()[1], output.size()[2], offset), out=None).cuda()
            output = torch.cat([output, addition], dim=3)

        out = torch.cat([input, output], dim=1)
        return out


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, scale_size):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_size, mode='bilinear',align_corners=True))

    def forward(self, input):
        return self.up(input)


class Encoder(tripleConv):
    def __init__(self, params):
        super(Encoder, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        # 每次tripleConv的输出
        out_block = super(Encoder, self).forward(input)

        # 是否使用dropout
        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        # 输出编码后的特征与索引数，方便后续拼接
        out_encoder, indices = self.maxpool(out_block)  # 下采样
        return out_encoder, out_block, indices