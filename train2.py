import os.path
import cv2
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from models import TopHat, ACM, get_segmentation_model, DNA_Net, EGPNet, UIUNet, MMLNet, MyModel
from argparse import ArgumentParser
from dataset_loader import dataset_loader
from metrics2 import SamplewiseSigmoidMetric, SmallTargetMetric, ROCMetric, PD_FA

# ACM损失函数
from loss import SoftLoULoss, BinaryDiceLoss

# 定义训练的超参数
def parse_args():
    parser = ArgumentParser(description = "Model hyperparameters")

    parser.add_argument("--model-name", type=str, default="MyModel", help = "the name of model")  # 模型名称
    parser.add_argument("--dataset", type = str, default=r".\dataset\NUAA", help = "the path of dataset")  # 数据集路径
    parser.add_argument("--save-path", type=str, default=r".\runs\NUAA", help="the path of results")  # 保存路径
    parser.add_argument("--weight-init", type=str, default=r"E:\project\SIRST20250220\runs\NUAA\MyModel\weights\MyModel_10 epoch.pth", help="kaiming or xavier")  # 初始化权重
    parser.add_argument("--start-from-0epoch", type=bool, default=True, help="start epoch")  # 有权重但是从头开始训练的epoch
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")  # 训练的epoch
    parser.add_argument("--save-epoch", type=int, default=10, help="save the model every n epochs")

    parser.add_argument("--batch-size", type=int, default = 4, help = "batch size")  # 训练时的batch size
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")  # 学习率
    parser.add_argument("--device", type=str, default="cuda", help="cpu or CUDA")  # 训练设备
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, default="train", help="train, val or test")
    parser.add_argument("--nclass", type=int, default=1, help="the number of classes")
    parser.add_argument("--score-thresh", type=float, default=0.4, help="the threshold of score")
    parser.add_argument("--bins", type=int, default=100, help="the number of bins")  # 计算ROC曲线的bins

    args = parser.parse_args()
    return args
class GradientMask(nn.Module):
    def __init__(self):
        super(GradientMask, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0


class Train:
    def __init__(self, args):
        """
            1. 创建数据集
            2. 数据集加载器
            3. 初始化模型参数
            4. GPU加速
            5. 损失函数，优化器初始化
            6. 保存路径
        """
        self.args = args
        self.train_dataset = dataset_loader(self.args.dataset, mode=self.args.mode)
        self.val_dataset = dataset_loader(self.args.dataset, mode="test")
        self.train_loader = Data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                            num_workers=8, drop_last=True)
        self.val_loader = Data.DataLoader(self.val_dataset, batch_size=1, num_workers=8)

        # self.model = TopHat(kernel_size=3)  # initialize the model
        # self.model = ACM([4, 4, 4], [8, 16, 32, 64], 'AsymBi')  # initialize the model
        # self.model = get_segmentation_model('agpcnet_1')
        # self.model = DNA_Net()
        # self.model = EGPNet()
        # self.model = UIUNet()
        # self.model = MMLNet()
        self.model = MyModel()

        if self.args.weight_init.endswith(".pth"):
            checkpoint = torch.load(self.args.weight_init)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = 1 if self.args.start_from_0epoch else checkpoint['epoch']
            # self.model.load_state_dict(torch.load(self.args.weight_init))
            print("权重加载成功！")
        else:
            self.start_epoch = 1
            self.model.apply(self.weight_init)
            print("无保存模型，将从头开始训练！")
        self.model.to(self.args.device)

        ##########################################################
        # LOSS
        self.criterion1 = SoftLoULoss()
        self.criterion2 = nn.BCEWithLogitsLoss()
        # self.criterion = BinaryDiceLoss()
        self.OhemCELoss = OhemCELoss()

        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.8, 0.99), eps=1e-04, weight_decay=self.args.weight_decay)

        ######################################################
        # optimizer
        optimizer_name = 'Adamweight'
        optimizer_settings = {"lr": self.args.lr}
        scheduler_name = 'CosineAnnealingLR'
        scheduler_settings = {'epochs': self.args.epochs, 'eta_min': self.args.min_lr, 'last_epoch': -1}
        self.optimizer, self.scheduler = get_optimizer(self.model, optimizer_name, scheduler_name, optimizer_settings,
                                                       scheduler_settings)

        # self.loss_list = []  # loss
        self.nIOU_metric = SamplewiseSigmoidMetric(self.args.nclass, self.args.score_thresh)
        self.small_target_metric = SmallTargetMetric(self.args.nclass, self.args.score_thresh)
        self.roc_metric = ROCMetric(self.args.nclass, self.args.bins)
        self.pd_fa_metric = PD_FA(self.args.nclass, self.args.bins)

        self.nIOU_metric.reset()
        self.small_target_metric.reset()
        self.roc_metric.reset()
        self.pd_fa_metric.reset()

        self.best = 0  # 记录最佳综合评价指标
        self.best_dict = {"epoch": 0, "avg_loss": 0, "P": 0, "R": 0, "nIOU": 0, "F1": 0, "Fa1": 0, "Fa2": [0], "OA": 0}
        self.results_list = []  # 存储所有训练和验证指标

        # save path
        self.results_path = os.path.join(self.args.save_path, self.args.model_name, "train-results.csv")
        # 获取目录路径（去掉文件名部分）
        dir_path = os.path.dirname(self.results_path)
        # 如果目录不存在，则创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.weights_path = os.path.join(self.args.save_path, self.args.model_name, "weights")  # 模型权重保存路径
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)

        # self.gradmask = GradientMask()

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.model.train()
            tbar = tqdm(self.train_loader)
            epoch_loss = []

            for i, data in enumerate(tbar):
                img, mask = data['img'].to(self.args.device), data['mask'].to(self.args.device)
                self.optimizer.zero_grad()

                output = self.model(img)
                # edge_gt = self.gradmask(mask)  # 边缘标签

                # 计算损失
                loss1 = self.criterion1(output, mask)
                # loss2 = self.criterion2(output, mask)

                # 损失求和
                # loss = 0.5 * loss1 + 0.5 * loss2
                loss = loss1

                epoch_loss.append(loss.item())

                # 保存输出图像
                for j in range(output.shape[0]):
                    img_np = ((output[j] > self.args.score_thresh).float().cpu().permute(1, 2,
                                                                                         0).detach().numpy() * 255).astype(
                        np.uint8)  # Convert to HWC format
                    save_imgs_path = os.path.join(self.args.save_path, self.args.model_name, "results-imgs-train",
                                                  "%s" % data["name"][j])
                    # 获取目录路径（去掉文件名部分）
                    dir_path = os.path.dirname(save_imgs_path)
                    # 如果目录不存在，则创建
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    cv2.imwrite(save_imgs_path, img_np)

                if (i + 1) % 8 == 0:
                    loss.backward()
                    self.optimizer.step()

                tbar.set_description("TRAIN: -->  Epoch %d, loss = %1.5f" % (epoch, epoch_loss[i]))

            ##############################################################
            # 调度器
            self.scheduler.step()

            avg_loss = np.mean(epoch_loss)
            print("TRAIN: -->  Epoch %d, loss = %1.5f" % (epoch, avg_loss))
            # self.val(epoch, avg_loss)

            # 保存模型权重
            if epoch % self.args.save_epoch == 0:  # 修改：每个10个epoch保存一次
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict()
                }
                torch.save(checkpoint,
                           os.path.join(self.weights_path, self.args.model_name + "_%d epoch.pth" % (epoch)))
            elif epoch == self.args.epochs:  # 保存最后一个epoch的模型权重
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict()
                }
                torch.save(checkpoint, os.path.join(self.weights_path, "last.pth"))

    def val(self, epoch, avg_loss):
        self.model.eval()
        tbar = tqdm(self.val_loader)
        for i, data in enumerate(tbar):
            img, mask = data['img'].to(self.args.device), data['mask'].to(self.args.device)
            with torch.no_grad():
                output = self.model(img)
                # output, gradient, gary = self.model(img)

                # 将pred替换为 二值元素
                # output = self.normPRED(output)
                # x = torch.zeros_like(output)
                # y = torch.ones_like(output)
                # output = torch.where(output >= 0.5, y, x)

                self.nIOU_metric.update(output, mask)
                self.small_target_metric.update(output, mask)
                self.roc_metric.update(output, mask)
                self.pd_fa_metric.update(output, mask)

        # 计算所有指标
        _, nIOU = self.nIOU_metric.get()
        P, R, Fa1, F1 = self.small_target_metric.get()
        Fa2, Pd, final_nIOU = self.pd_fa_metric.get(len(self.val_dataset))
        TPR, FPR = self.roc_metric.get()
        overall_accuracy = nIOU + P + R + F1 - Fa1
        if overall_accuracy > self.best:
            self.best = overall_accuracy  # 更新最佳模型性能
            self.best_dict = {"epoch": epoch, "avg_loss": avg_loss, "P": P, "R": R, "nIOU": nIOU, "F1": F1, "Fa1": Fa1,
                              "Fa2": Fa2, "OA": overall_accuracy}
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict()
            }
            torch.save(checkpoint, os.path.join(self.weights_path, "best.pth"))
        f = "The best model metrics: --> Epoch: %d, loss: %1.5f, nIOU: %1.5f, P: %1.5f, R: %1.5f, Fa1: %1.5f, Fa2: %1.5f, F1: %1.5f, OA: %1.5f"
        print(f % (self.best_dict["epoch"], self.best_dict["avg_loss"], self.best_dict["nIOU"], self.best_dict["P"],
                   self.best_dict["R"],
                   self.best_dict["Fa1"], self.best_dict["Fa2"][0] * 1000000, self.best_dict["F1"],
                   self.best_dict["OA"]))

        # 记录数据列表
        self.results_list.append([epoch, avg_loss, nIOU, P, R, Fa1, Fa2, F1, TPR, FPR, overall_accuracy])
        self.save_metrics()
        s = "current model metrics: --> Epoch: %d, loss: %1.5f, nIOU: %1.5f, P: %1.5f, R: %1.5f, Fa1: %1.5f, Fa2: %1.5f, F1: %1.5f, OA: %1.5f"
        print(s % (epoch, avg_loss, nIOU, P, R, Fa1, Fa2[0] * 1000000, F1, overall_accuracy))


    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # nn.init.constant_(m.weight, 1)
            # nn.init.constant_(m.bias, 0)
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)

    def save_metrics(self):
        """使用 pandas 保存训练数据到 CSV"""
        df = pd.DataFrame(self.results_list, columns=["Epoch", "Loss", "nIOU", "P", "R", "Fa1", "Fa2", "F1", "TPR", "FPR", "overall_accuracy"])
        df.to_csv(self.results_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    train = Train(args)
    train.train()
    # for data in train.train_dataset:
    #     img = data["img"]
    #     mask = data["mask"]
    #     print(img.shape)
    #     print(mask.shape)
        # if img.shape[2] != 3:
        #     print(img.shape)
        #     print(data["name"])