import torch
import numpy as np
import my_utils
import time
from tqdm import tqdm


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """
    """
    model: 模型
    nepochs: 训练迭代次数
    sbatch: 每个批次的样本数
    lr: 学习率
    lr_min: 最小学习率
    lr_factor: 学习率减小因子
    lr_patience: 容忍验证集性能不能提高的周期数，在周期内性能没有提升则调整学习率
    clipgrade: 梯度裁剪的阈值，防止梯度爆炸
    lamb: 正则化项，损失函数中正则化项的系数
    args: 允许外部参数传递给类的初始化函数
    """

    def __init__(self, model, nepochs=100, sbatch=64, lr=0.05, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=100,
                 lamb=1000):
        self.model = model
        self.fisher = None  # 存储Fisher信息矩阵值

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()  # 计算交叉熵损失
        self.optimizer = self._get_optimizer()
        self.lamb = lamb
        self.old_num_class = None
        self.num_class = None
        self.old_parameter = None

        return

    # 创建优化器
    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=lr, weight_decay=0.0005)  # 创建一个随机梯度下降优化器，对模型的参数进行更新

    # 进行训练
    def train(self, t, xtrain, ytrain, xvalid, yvalid, taskcla):
        """
        t: 第t个任务
        xtrain: 训练数据特征
        ytrain: 训练数据标签
        xvalid: 验证数据特征
        yvalid: 验证数据标签
        """
        self.old_num_class = t * taskcla[0][1]
        self.num_class = (t + 1) * taskcla[0][1]
        if t > 0:
            # 更新模型
            self.model.update_fc(self.num_class)

        best_loss = np.inf  # 初始化最佳损失
        best_model = my_utils.get_model(self.model)  # 初始化最佳模型
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # 迭代训练
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()  # 训练开始前实践
            self.train_epoch(t, xtrain, ytrain)  # 一个迭代轮次训练
            clock1 = time.time()  # 训练结束的时间
            train_loss, train_acc = self.eval(t, xtrain, ytrain)  # 评估训练损失和准确率
            clock2 = time.time()  # 评估结束时间
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0),
                1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0), train_loss, 100 * train_acc), end='')
            # 评估验证集上的损失和正确率
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            # 更新最佳损失和模型参数
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = my_utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                # 如果达到了忍耐周期，则调整学习率
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    # 当学习率小于设定的最小学习率时，停止训练
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # 恢复到最佳模型参数
        my_utils.set_model_(self.model, best_model)

        # 计算当前任务的fisher信息矩阵并与历史fisher信息矩阵合并
        if self.fisher is None:
            self.fisher = self.getFisherDiagonal(xtrain, ytrain)
        else:
            alpha = self.old_num_class / self.num_class
            new_fisher = self.getFisherDiagonal(xtrain, ytrain)
            for n, p in new_fisher.items():
                new_fisher[n][: len(self.fisher[n])] = (
                        alpha * self.fisher[n]
                        + (1 - alpha) * new_fisher[n][: len(self.fisher[n])]
                )
            self.fisher = new_fisher
        # 保存当前模型的参数
        self.old_parameter = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        return

    # 每轮迭代的训练
    def train_epoch(self, t, x, y):
        self.model.train()  # 设置模型为训练模式

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # 循环每个批次训练
        for i in range(0, len(r), self.sbatch):
            # 获取数据索引
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            # 获取索引的数据特征和标签
            images = x[b]
            targets = y[b]

            # 对模型进行前向传播
            output = self.model.forward(images)
            loss = self.criterion(t, output[:, self.old_num_class:], targets - self.old_num_class)

            # 后向传播更新模型参数
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        return

    # 对模型进行评估
    def eval(self, t, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()  # 将模型设置为评估模式

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = x[b]
            targets = y[b]

            # 前向传播
            output = self.model.forward(images)
            loss = self.criterion(t, output, targets)

            pred = torch.max(output, dim=1)[1]  # 将预测概率最大作为预测标签
            hits = (pred == targets).float()

            # Log
            total_loss += loss.item() * len(b)
            total_acc += hits.sum().item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    # 计算总的损失
    def criterion(self, t, output, targets):
        loss_reg = 0
        if t > 0:
            # 计算正则化部分的损失
            for name, param in self.model.named_parameters():
                if name in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[name]
                                          * (self.old_parameter[name] - param[:len(self.old_parameter[name])]).pow(2)) / 2

        return self.ce(output, targets) + self.lamb * loss_reg

    # 计算fisher信息矩阵
    def getFisherDiagonal(self, x, y, sbatch=20):
        # 初始化fisher信息矩阵
        fisher = {
            n: torch.zeros(p.shape).cuda()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.model.train()
        for i in tqdm(range(0, x.size(0), sbatch), desc='Fisher diagonal', ncols=100, ascii=True):
            b = torch.LongTensor(np.arange(i, np.min([i + sbatch, x.size(0)]))).cuda()
            images = x[b]
            target = y[b]
            self.model.zero_grad()
            output = self.model(images)
            loss = self.ce(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += sbatch * p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(x)
            fisher[n] = torch.min(fisher[n], torch.tensor(0.0001))

        return fisher
