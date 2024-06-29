import argparse
import time
import numpy as np
import torch
import sys
import my_utils

tstart = time.time()

# 输入参数
parser = argparse.ArgumentParser(description='EWC')
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')  # 设置随机种子
parser.add_argument('--dataset', default='', type=str, required=True,
                    choices=['tasks_5', 'tasks_10', 'tasks_20'],
                    help='(default=%(default)s)')  # 设置数据集,对应三种不同的训练策略
parser.add_argument('--approach', default='', type=str, required=True,
                    choices=['ewc', 'ewc_plus', 'none_ewc'], help='(default=%(default)s)')  # 设置算法
parser.add_argument('--network', default='sim_network', type=str, required=True,
                    choices=['sim_alexnet', 'alexnet'], help='(default=%(default)s)')  # 设置网络模型
parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')  # 输出路径
parser.add_argument('--nepochs', default=200, type=int, required=False, help='(default=%(default)d)')  # 迭代轮次
parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')  # 初始学习率
parser.add_argument('--lamb', default=500, type=int, required=False, help='(default=%(default)d)')  # 正则化强度
args = parser.parse_args()
if args.output == '':
    args.output = 'output/' + args.dataset + '_' + args.approach + '_' + args.network + '_' + str(args.lamb) + '.txt'  # 默认输出路径
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# 检查GPU是否可用
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()

# 调用相应的增量策略
if args.dataset == 'tasks_5':
    from dataloaders import tasks_5 as dataloader
elif args.dataset == 'tasks_10':
    from dataloaders import tasks_10 as dataloader
elif args.dataset == 'tasks_20':
    from dataloaders import tasks_20 as dataloader

# 调用相应的增量学习算法
if args.approach == 'ewc':
    from approaches import ewc as approach
elif args.approach == 'ewc_plus':
    from approaches import ewc_plus as approach
elif args.approach == 'none_ewc':
    from approaches import none_ewc as approach

# 调用相应的神经网络
if args.network == 'sim_alexnet':
    from networks import sim_alexnet as network
elif args.network == 'alexnet':
    from networks import AlexNet as network


# 加载数据
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)
# 每个任务中的类别数量
per_class = taskcla[0][1]

# 初始化网络
print('Inits...')
net = network.Net(per_class).cuda()

# 初始化算法训练参数
appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, lamb=args.lamb)
print(appr.criterion)
my_utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

# 任务循环
acc = np.zeros((len(taskcla), (len(taskcla)) + 1), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # 获取训练集和验证集
    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['valid']['x'].cuda()
    yvalid = data[t]['valid']['y'].cuda()
    task = t

    # 标签偏移
    ytrain += t * ncla
    yvalid += t * ncla
    # 训练
    appr.train(task, xtrain, ytrain, xvalid, yvalid, taskcla)
    print('-' * 100)

    # 对每个任务进行验证
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        ytest += u * ncla
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    mean_acc = 0
    print('mean_Accuracies=')
    print('\t', end='')
    for j in range(t + 1):
        mean_acc += acc[t, j]
    mean_acc = mean_acc / (t + 1)
    print(mean_acc)
    acc[t, len(taskcla)] = mean_acc
    # 保存正确率结果
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')

print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
mean_acc = 0
num_task = len(taskcla)
print('mean_Accuracies=')
print('\t', end='')
for j in range(num_task):
    mean_acc += acc[num_task - 1, j]
print(mean_acc / num_task)
print('结束!')

# 输出总花费时间
print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
