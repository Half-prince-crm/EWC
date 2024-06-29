import numpy as np
from copy import deepcopy


# 返回模型参数集合的拷贝
def get_model(model):
    return deepcopy(model.state_dict())


# 更新设置模型的参数
def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


# 冻结模型参数，不进行更新
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

# 将数据转为更易读的格式
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# 计算卷积操作后的输出尺寸
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))

# 打印优化器非参数配置信息
def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim, '=', end=' ')
        opt = optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()
    return
