import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class TopHatTransform(nn.Module):
    def __init__(self, kernel_size=3, device=None):
        super(TopHatTransform, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.kernel_size = kernel_size

    def morphological_opening(self, images):
        """
        Perform morphological opening (erosion followed by dilation) using the predefined kernel.
        """
        B, C, H, W = images.shape
        kernel = torch.ones((C, 1, self.kernel_size, self.kernel_size), dtype=torch.float32, device=self.device) / (
                    self.kernel_size ** 2)

        padding = self.kernel_size // 2
        eroded = F.conv2d(images, kernel, padding=padding, groups=C)
        dilated = F.conv2d(eroded, kernel, padding=padding, groups=C)
        return dilated

    def forward(self, images):
        """
        Compute the top-hat transformation: images - opened_images.
        """
        opened_images = self.morphological_opening(images)
        return images - opened_images

if __name__ == "__main__":
    latency = []
    for i in range(100):
        # 生成随机输入数据
        input_data = torch.randn(1, 3, 1024, 1024).cuda()  # 示例输入数据
        # 计算Latency
        model = TopHatTransform(kernel_size=3)
        model.eval()
        start_time = time.time()
        result = model(input_data)
        print(result.shape)
        latency.append((time.time() - start_time) * 1000)  # 毫秒
    print(f"Latency: {sum(latency)/len(latency):.2f} ms")

    # # 计算Latency
    # model = TopHatTransform(kernel_size=3)
    # model.eval()
    # start_time = time.time()
    # result = model(input_data)
    # latency = (time.time() - start_time) * 1000  # 毫秒
    # print(f"Latency: {latency:.2f} ms")

    # 计算FLOPs
    input_data = torch.randn(1, 1, 1024, 1024)  # 示例输入数据
    macs, params = profile(model, inputs=(input_data,))
    FLOPs = 2 * macs
    print(f"FLOPs: {FLOPs / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
    print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）