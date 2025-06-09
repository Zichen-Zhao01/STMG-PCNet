import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss

class OhemCELoss(nn.Module):
    '''
    在计算完所有像素级交叉熵后，只选择“最难”那部分样本（具有最高的 loss）来参与梯度更新，
    从而更加聚焦于对模型提升最有价值的 hard examples。这正是 OHEM 的精髓——自动挑选难例，
    减少易例对训练的冗余影响

    thresh：难例阈值
    代码中将用户给定的 thresh（通常是一个概率值，如 0.7）通过 -log(thresh) 转换为对应的 交叉熵损失下限。
    若某像素的交叉熵 大于 这个下限，则认为它是 hard sample。
    n_min：最少保留数
    即使所有像素的 loss 都小于阈值，也至少 保留前 n_min 个最大的 loss 样本，保证梯度不为空。
    '''
    def __init__(self, thresh=0.7, n_min=16*512*512//16, ignore_lb=1, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        #  将概率阈值 t 转换到对数空间作为 loss 下限
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min   # 最少保留样本数
        self.ignore_lb = ignore_lb  # 忽略标签
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')  # 不立即求均值

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        # 1. 计算每像素交叉熵，展平成一维向量
        loss = self.criteria(logits, labels).view(-1)
        # 2. 将所有像素 loss 从大到小排序
        loss, _ = torch.sort(loss, descending=True)
        # 3. 如果第 n_min 大的 loss 超过阈值，则舍弃较易样本，只保留 loss > thresh
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            # 否则至少保留前 n_min 个 hardest examples
            loss = loss[:self.n_min]
        # 4. 对保留下来的 hardest loss 求平均
        return torch.mean(loss)

class OhemCELoss(nn.Module):
    def __init__(self, thresh=0.7, n_min=16*512*512//16, ignore_lb=1, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh_val = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')  # 正确设置ignore_index

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        mask = labels.view(-1) != self.ignore_lb
        valid_loss = loss[mask]

        if valid_loss.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        thresh = -torch.log(torch.tensor(self.thresh_val, dtype=torch.float, device=logits.device))
        sorted_loss, _ = torch.sort(valid_loss, descending=True)
        n_min = min(self.n_min, sorted_loss.numel())

        if n_min == 0:
            return torch.mean(sorted_loss) if sorted_loss.numel() > 0 else torch.tensor(0.0, device=logits.device)

        if sorted_loss[n_min - 1] > thresh:
            keep_loss = sorted_loss[sorted_loss > thresh]
        else:
            keep_loss = sorted_loss[:n_min]

        return torch.mean(keep_loss) if keep_loss.numel() > 0 else torch.tensor(0.0, device=logits.device)

if __name__ == '__main__':
    # 示例输入
    logits = torch.randn(2, 1, 512, 512)  # 假设是一个2x3x5x5的logits张量
    labels = torch.randn(2, 1, 512, 512)  # 假设是一个2x5x5的标签张量
    # 创建OHEM损失实例
    ohem_loss = OhemCELoss(thresh=0.7, n_min=10)
    # 计算OHEM损失
    loss = ohem_loss(logits, labels)
    print("OHEM Loss:", loss)
    # 计算交叉熵损失
    cross_entropy_loss = nn.CrossEntropyLoss()(logits, labels)
    print("Cross Entropy Loss:", cross_entropy_loss)
