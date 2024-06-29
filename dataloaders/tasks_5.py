import os
import torch
import numpy as np
from sklearn.utils import shuffle
from torchvision import datasets, transforms

def get(seed=0, pc_valid=0.10):
    """
    seed: 随机种子
    pc_valid: 验证集比例
    """
    data = {}
    taskcla = []
    size = [3, 32, 32]

    if not os.path.isdir('dataset_5/pre_cifar/'):
        os.makedirs('dataset_5/pre_cifar')

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        # 进行数据增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
            transforms.RandomHorizontalFlip(),  # 随机翻转

            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean, std),  # 标准化
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 加载和预处理CIFAR100数据集
        dat = {'train': datasets.CIFAR100('dataset_5/', train=True, download=True, transform=transform_train),
               'test': datasets.CIFAR100('dataset_5/', train=False, download=True, transform=transform_test)}

        # 将100类图片分为20类一组的5个分类任务
        for n in range(5):
            data[n] = {}
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 20
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}

        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = n // 20
                data[nn][s]['x'].append(image)
                data[nn][s]['y'].append(n % 20)

        # 格式化和保存
        for t in data.keys():
            for s in ['train', 'test']:
                # 对数据进行格式化处理
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                # 将数据保存为二进制格式文件
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('dataset_5/pre_cifar'),
                                        'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('dataset_5/pre_cifar'),
                                        'data' + str(t) + s + 'y.bin'))
    # 从二进制文件中加载数据
    data = {}
    ids = list(shuffle(np.arange(5), random_state=seed))
    print('Task order =', ids)
    for i in range(5):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('dataset_5/pre_cifar'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('dataset_5/pre_cifar'), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100-' + str(ids[i])

    # 划分验证集
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    """
    data: 处理后的数据集
    taskcla: 任务标号及对应类别数量
    size: 单张图片的尺寸
    """
    return data, taskcla, size
