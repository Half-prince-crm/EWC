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

    if not os.path.isdir('dataset_20/pre_cifar/'):
        os.makedirs('dataset_20/pre_cifar')

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        # 进行数据增强
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机翻转
            transforms.RandomRotation(15),  # 随机旋转
            transforms.ColorJitter(brightness=0.2,  # 颜色抖动
                                   contrast=0.2,
                                   saturation=0.2,
                                   hue=0.1),
            transforms.RandomAffine(degrees=0,  # 平移和缩放
                                    translate=(0.1, 0.1),
                                    scale=(0.8, 1.2)),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean, std),  # 标准化
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 加载和预处理CIFAR100数据集
        dat = {'train': datasets.CIFAR100('dataset_20/', train=True, download=True, transform=transform_train),
               'test': datasets.CIFAR100('dataset_20/', train=False, download=True, transform=transform_test)}

        # 将100类图片分为5类一组的20个分类任务
        for n in range(20):
            data[n] = {}
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 5
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}

        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = n // 5
                data[nn][s]['x'].append(image)
                data[nn][s]['y'].append(n % 5)

        # 格式化和保存
        for t in data.keys():
            for s in ['train', 'test']:
                # 对数据进行格式化处理
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                # 将数据保存为二进制格式文件
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('dataset_20/pre_cifar'),
                                        'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('dataset_20/pre_cifar'),
                                        'data' + str(t) + s + 'y.bin'))
    # 从二进制文件中加载数据
    data = {}
    ids = list(shuffle(np.arange(20), random_state=seed))
    ids[14] = 0
    ids[0] = 4
    print('Task order =', ids)
    for i in range(20):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('dataset_20/pre_cifar'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('dataset_20/pre_cifar'), 'data' + str(ids[i]) + s + 'y.bin'))
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
