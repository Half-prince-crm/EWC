import torch
import copy
import torch.nn as nn
import my_utils
from torch.nn import functional as F

class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Net(torch.nn.Module):

    def __init__(self, num_class):
        super(Net, self).__init__()

        s = my_utils.compute_conv_output_size(32, 4)
        s = s // 2
        s = my_utils.compute_conv_output_size(s, 3)
        s = s // 2
        s = my_utils.compute_conv_output_size(s, 2)
        s = s // 2
        # 特征提取，卷积
        self.features_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(128, 256, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # 特征提取， 全连接
        self.features_fc = torch.nn.Sequential(
            torch.nn.Linear(256 * s * s, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )

        # 分类层
        self.in_features = 2048
        self.fc = nn.Linear(self.in_features, num_class)
        return

    # 前向传播
    def forward(self, x):
        h = self.features_conv(x)
        # 展成一维向量，作为全连接层的输入
        h = h.view(x.size(0), -1)
        h = self.features_fc(h)
        y = self.fc(h)
        return y

    def update_fc(self, num_class):
        fc_new = SimpleLinear(self.in_features, num_class).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc_new.weight.data[:nb_output] = weight
            fc_new.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc_new
