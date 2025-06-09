import cv2
import os
import time
import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm
from thop import profile
from test_model import test_model
from models import TopHat, ACM, get_segmentation_model, DNA_Net, EGPNet, MyModel
from argparse import ArgumentParser
from dataset_loader import dataset_loader
from metrics2 import SamplewiseSigmoidMetric, SmallTargetMetric, ROCMetric, PD_FA

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="MyModel", help="the name of model")
    parser.add_argument("--dataset", type=str, default=r"./dataset/NUAA", help='the path of dataset')  # 数据集路径
    parser.add_argument("--weight-path", type=str, default=r'./runs/NUAA/MyModel/weights/MyModel_10 epoch.pth', help="the path of model weight, .pth")
    parser.add_argument("--save-path", type=str, default=r".\runs\NUAA", help="the path of results")  # 运行结果路径
    parser.add_argument("--mode", type=str, default="val", help="val or test")

    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--nclass", type=int, default=1, help="the number of classes")
    parser.add_argument("--score-thresh", type=float, default=0.4, help="the threshold of score")
    parser.add_argument("--bins", type=int, default=100, help="the number of bins")  # 计算ROC曲线的bins


    args = parser.parse_args()
    return args

class Test:
    def __init__(self, args):
        self.args = args
        self.test_dataset = dataset_loader(self.args.dataset, mode=self.args.mode)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=1)
        # self.model = ACM([4, 4, 4], [8, 16, 32, 64], 'AsymBi')
        # self.model = get_segmentation_model('agpcnet_1')
        # self.model = DNA_Net()
        # self.model = EGPNet()
        # self.model = TopHat(kernel_size=3, device="cuda:0")
        self.model = MyModel()


        # 加载模型权重
        # if os.path.exists(args.weight_path) and args.weight_path.endswith(".pth"):
        #     # self.model.load_state_dict(torch.load(args.weight_path))
        #     checkpoint = torch.load(self.args.weight_path)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     print("权重加载成功！")
        # else:
        #     raise ValueError("The weight file does not exist or is not a .pth file.")
        self.model.to(self.args.device)

        # 初始化评价指标
        self.nIOU_metric = SamplewiseSigmoidMetric(self.args.nclass, self.args.score_thresh)
        self.small_target_metric = SmallTargetMetric(self.args.nclass, self.args.score_thresh)
        self.roc_metric = ROCMetric(self.args.nclass, self.args.bins)
        self.pd_fa_metric = PD_FA(self.args.nclass, self.args.bins)

        self.nIOU_metric.reset()
        self.small_target_metric.reset()
        self.roc_metric.reset()
        self.pd_fa_metric.reset()

        self.results_list = []  # 存储所有训练和验证指标

        # save path
        self.results_path = os.path.join(self.args.save_path, self.args.model_name, self.args.mode + "-results.csv")
        # 获取目录路径（去掉文件名部分）
        dir_path = os.path.dirname(self.results_path)
        # 如果目录不存在，则创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def test(self):
        self.model.eval()
        tbar = tqdm(self.test_loader)
        latency = []  # 记录延迟指标
        flops = []  # 记录FLOPs
        paramter = []  # 记录参数量
        for i, data in enumerate(tbar):
            img, mask = data['img'].to(self.args.device), data['mask'].to(self.args.device)
            with torch.no_grad():
                start_time = time.time()
                # output = self.model(img[:, :1, ...])
                output = self.model(img)

                # output的二值化
                output = self.normPRED(output)
                x = torch.zeros_like(output)
                y = torch.ones_like(output)
                output = torch.where(output > 0.55, y, x)
                latency.append((time.time() - start_time) * 1000)  # 毫秒

                # 计算参数量和计算量
                macs, params = profile(self.model, inputs=(img,))
                FLOPs = 2 * macs
                flops.append(FLOPs)
                paramter.append(params)


                for j in range(output.shape[0]):
                    img_np = ((output[j] > self.args.score_thresh).float().cpu().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)  # Convert to HWC format
                    save_imgs_path = os.path.join(self.args.save_path, self.args.model_name, "results-imgs-" + self.args.mode, "%s" % data["name"][j])
                    # 获取目录路径（去掉文件名部分）
                    dir_path = os.path.dirname(save_imgs_path)
                    # 如果目录不存在，则创建
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    cv2.imwrite(save_imgs_path, img_np)

                # 计算所有指标
                self.nIOU_metric.update(output, mask)
                self.small_target_metric.update(output, mask)
                self.roc_metric.update(output, mask)
                self.pd_fa_metric.update(output, mask)

        # 计算延迟指标
        print(f"Latency: {sum(latency) / len(latency):.2f} ms")
        # 计算FLOPs和参数量
        print(f"FLOPs: {sum(flops)/len(flops) / 1e9:.2f} G")
        print(f"Params: {sum(paramter)/len(paramter) / 1e6:.2f} M")

        _, nIOU = self.nIOU_metric.get()
        P, R, Fa1, F1 = self.small_target_metric.get()
        Fa2, Pd, final_nIOU = self.pd_fa_metric.get(len(self.test_dataset))
        TPR, FPR = self.roc_metric.get()
        overall_accuracy = nIOU + P + R + F1 - Fa1
        self.results_list.append([nIOU, P, R, Fa1, Fa2, Pd, F1, TPR, FPR, overall_accuracy])
        self.save_metrics()
        print("final_nIOU:",final_nIOU)
        print(
            "current model metrics: --> nIOU: %1.5f, P: %1.5f, R: %1.5f, Fa1: %1.5f, Fa2: %1.5f, Pd: %1.5f, F1: %1.5f, OA: %1.5f" % (
            nIOU, P, R, Fa1, Fa2[0]*1000000, Pd[0], F1, overall_accuracy))

    def save_metrics(self):
        """使用 pandas 保存训练数据到 CSV"""
        df = pd.DataFrame(self.results_list, columns=["nIOU", "P", "R", "Fa1", "Fa2", "Pd", "F1", "TPR", "FPR", "overall_accuracy"])
        df.to_csv(self.results_path, index=False)

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn


if __name__ == "__main__":
    args = parse_args()
    test = Test(args)
    test.test()