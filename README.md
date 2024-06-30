下载以后使用python在终端运行run.py，指定对应的分类数据集dataset（task_5、task_10、task_20）,对应的增量算法approach（ewc、ewc_plus、none）,对应的神经网络（AlexNet、sim_alexnet），即可运行

例子："python run.py --dataset task_5 --approach ewc  --Alexnet"  --> 用 5task*20classes 的增量方式,在alex网络上进行EWC算法

初次使用时要下载数据集，速度较慢；第一次以后将变快
