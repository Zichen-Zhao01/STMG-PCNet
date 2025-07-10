import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out

        
class ENDHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ENDHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)
        

class LTA(nn.Module):
    def __init__(self, ksize=3):
        super(LTA, self).__init__()
        self.ksize = ksize
      
    def forward(self, img):
        # pdb.set_trace()
        B, C, H, W = img.shape
        pad = (self.ksize - 1) // 2
        img = F.pad(img, [pad, pad, pad, pad], mode='constant', value=0)
  
        patches = img.unfold(dimension=2, size=self.ksize, step=1)
        patches = patches.unfold(dimension=3, size=self.ksize, step=1)

        newImg, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)

        return newImg

class FilterLayer(nn.Module):
   def __init__(self, in_planes, out_planes, reduction=16):
       super(FilterLayer, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.fc = nn.Sequential(
           nn.Linear(in_planes, out_planes // reduction),
           nn.ReLU(inplace=True),
           nn.Linear(out_planes // reduction, out_planes),
           nn.Sigmoid()
       )
       self.out_planes = out_planes
   def forward(self, x):
       b, c, _, _ = x.size()
       y = self.avg_pool(x).view(b, c)
       y = self.fc(y).view(b, self.out_planes, 1, 1)
       return y

class FSP(nn.Module):
   def __init__(self, in_planes, out_planes, reduction=16):
       super(FSP, self).__init__()
       self.filter = FilterLayer(2*in_planes, out_planes, reduction)
   def forward(self, guidePath, mainPath):
       combined = torch.cat((guidePath, mainPath), dim=1)
       channel_weight = self.filter(combined)
       out = mainPath + channel_weight * guidePath
       return out


class RIRM(nn.Module):
    def __init__(self):
        super(RIRM, self).__init__()
        self.ccsolver = CCSolver()

    def forward(self, prob, feat):

        B,C,H,W = prob.shape
        r = 2
        feats = []
        
        for i in range(B):
            
            labels = self.ccsolver(prob[i].detach()) 
            props = measure.regionprops(labels)
            confidence_values = prob[i].detach().flatten().cpu().numpy()

            connected_regions = []
            confidences = []

            if len(np.unique(labels)) - 1 == 0:
                
                feature = F.interpolate(feat[i].unsqueeze(0), size=(H//r, W//r))
                feats.append(torch.cat([feature,feature,feature],dim=1))
            else:
                region_labels = np.unique(labels)[1:]
                for region_label in region_labels:
                    region_pixels = (labels == region_label).flatten()
                    region_confidences = confidence_values[region_pixels]
                    confidence_mean = np.mean(region_confidences)
                    confidences.append(confidence_mean)
                
                sorted_indices = np.argsort(confidences)[::-1]
                for index in sorted_indices:
                    region_label = region_labels[index]
                    if region_label not in connected_regions:
                        connected_regions.append(region_label)
                        if len(connected_regions) == min(3, len(np.unique(labels)) - 1):
                            break
      
                feature = []
                for region_label in connected_regions:
                    min_row, min_col, max_row, max_col = props[region_label-1].bbox
                    cropped_feature = feat[i][None, :, min_row:max_row, min_col:max_col]
                    cropped_feature = F.interpolate(cropped_feature, size=(H//r, W//r))
                    feature.append(cropped_feature)
                if len(connected_regions) == 1:
                    feature = [feature[0], feature[0], feature[0]]
                elif len(connected_regions) == 2:
                    feature = [feature[0], feature[0], feature[1]]
                
                feats.append(torch.cat(feature, dim=1))
        feats = torch.cat(feats, dim=0)
        return feats

class CCSolver(nn.Module):
    def __init__(self):
        super(CCSolver, self).__init__()

    def forward(self, prob):
        binary_image = (prob > 0.5).float() 
        binary_image = binary_image.squeeze().cpu().numpy()  
        labels = measure.label(binary_image, connectivity=2)
        return labels

class Resblock(nn.Module):
    def __init__(self,  n_feats, kernel_size = 3, padding = 1, bias = False, act=nn.ReLU(inplace=True)):
        super(Resblock, self).__init__()        
        
        m = []
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size = kernel_size, padding = padding, bias=bias))
        m.append(act)
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size = kernel_size, padding = padding, bias=bias))
   
        self.m = nn.Sequential(*m)
    
    def forward(self, x):
        
        x = self.m(x) + x    
        return x 

class EGPNet(nn.Module):
    def __init__(self, layer_blocks, channels):
        super(EGPNet, self).__init__()
 
        self.pool  = nn.MaxPool2d(2, 2)  
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

        self.stage1 = self._make_layer(block=ResidualBlock, block_num=2,
                                        in_channels=7, out_channels=channels[0], stride=1)
        self.stage1_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0], out_channels=channels[0], stride=1)
        self.stage1_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0]*2, out_channels=channels[0]*2, stride=1)
        self.stage2_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0], out_channels=channels[0], stride=1)
        self.stage2_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0]*2, out_channels=channels[0]*2, stride=1)


        self.stage2_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1], out_channels=channels[1], stride=1)
        self.stage3_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1], out_channels=channels[1], stride=1)
        self.stage3_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1]*2, out_channels=channels[1]*2, stride=1)
        self.stage2_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1]*2, out_channels=channels[1]*2, stride=1)


        self.stage3_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2], out_channels=channels[2], stride=1)
        self.stage4_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2], out_channels=channels[2], stride=1)
        self.stage4_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2]*2, out_channels=channels[2]*2, stride=1)
        self.stage3_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2]*2, out_channels=channels[2]*2, stride=1)


        self.edge_stage1 = self._make_layer(block=ResidualBlock, block_num=2,
                                        in_channels=3, out_channels=channels[0], stride=1)
        self.edge_stage1_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0], out_channels=channels[0], stride=1)
        self.edge_stage1_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0]*2, out_channels=channels[0]*2, stride=1)
        self.edge_stage2_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0], out_channels=channels[0], stride=1)
        self.edge_stage2_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[0]*2, out_channels=channels[0]*2, stride=1)


        self.edge_stage2_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1], out_channels=channels[1], stride=1)
        self.edge_stage3_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1], out_channels=channels[1], stride=1)
        self.edge_stage3_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1]*2, out_channels=channels[1]*2, stride=1)
        self.edge_stage2_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[1]*2, out_channels=channels[1]*2, stride=1)


        self.edge_stage3_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2], out_channels=channels[2], stride=1)
        self.edge_stage4_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2], out_channels=channels[2], stride=1)
        self.edge_stage4_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2]*2, out_channels=channels[2]*2, stride=1)
        self.edge_stage3_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                        in_channels=channels[2]*2, out_channels=channels[2]*2, stride=1)


        self.uplayer2 = self._make_layer(block=ResidualBlock, block_num=2,
                                         in_channels=channels[3], out_channels=channels[3], stride=1)
        self.uplayer2_edge = self._make_layer(block=ResidualBlock, block_num=2,
                                         in_channels=channels[3], out_channels=channels[3], stride=1)

        self.uplayer1 = self._make_layer(block=ResidualBlock, block_num=2,
                                         in_channels=channels[2], out_channels=channels[2], stride=1)
        self.uplayer1_edge = self._make_layer(block=ResidualBlock, block_num=2,
                                         in_channels=channels[2], out_channels=channels[2], stride=1)

        self.lta = LTA()                        

        sobel_x = torch.tensor([[[1, 0], [0, -1]], 
                               [[1, 0], [0, -1]], 
                               [[1, 0], [0, -1]]], dtype=torch.float32)
        sobel_x = sobel_x.reshape(1,3,2,2).cuda()
        sobel_y = torch.tensor([[[0, 1], [-1, 0]], 
                               [[0, 1], [-1, 0]], 
                               [[0, 1], [-1, 0]]], dtype=torch.float32)
        sobel_x = sobel_x.reshape(1,3,2,2).cuda() 
        sobel_y = sobel_y.reshape(1,3,2,2).cuda()        
        self.weight_x = nn.Parameter(data = sobel_x, requires_grad = False).cuda()
        self.weight_y = nn.Parameter(data = sobel_y, requires_grad = False).cuda()
                                               
        self.fsp_rgb = FSP(128, 128, reduction=16)
        self.fsp_hha = FSP(128, 128, reduction=16)
        self.fsp_rgb1 = FSP(64, 64, reduction=8)
        self.fsp_hha1 = FSP(64, 64, reduction=8)

        self.fuse = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)
        self.reduce2_edge = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.reduce1_edge = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.reduce2 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.reduce1 = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)

        self.eirm = RIRM()

        self.fuse_local = nn.Conv2d(96, 128, kernel_size=1, padding=0, bias=False)
        self.upsample = nn.PixelShuffle(2)

        self.resblock = Resblock(64)
        self.end = ENDHead(64, 1)

    def forward(self, x):

        _, _, hei, wid = x.shape 
        x_size = x.size()
        x_dilate = self.lta(x) 
 
        Gx = F.conv2d(x_dilate, self.weight_x, padding='same')
        Gy = F.conv2d(x_dilate, self.weight_y, padding='same') 
        x_gard = torch.sqrt(Gx**2 + Gy**2)

        stage1 = self.stage1(torch.cat([x, x_gard, x_dilate], 1))  # 16x480x480
        
        fisrt = self.stage1_1(stage1) # 16x480x480
           
        second = self.stage2_1(self.pool(stage1)) # 16x240x240
        third = self.stage2_2(torch.cat([self.down(fisrt), second],1)) # 32x240x240
        stage1_2 = self.stage1_2(torch.cat([self.up(second), fisrt],1)) # 32x480x480

        fisrt = self.stage2_3(third)   # 32x240x240
        second = self.stage3_1(self.pool(third))  # 32x120x120
        third = self.stage3_2(torch.cat([self.down(fisrt), second],1))  # 64x120x120
        stage2_4 = self.stage2_4(torch.cat([self.up(second), fisrt],1)) # 64x240x240

        fisrt = self.stage3_3(third)   # 64x120x120
        second = self.stage4_1(self.pool(third))  # 64x60x60
        stage4_2 = self.stage4_2(torch.cat([self.down(fisrt), second],1)) #128x60x60
        stage3_4 = self.stage3_4(torch.cat([self.up(second), fisrt],1)) # 128x120x120


        stage1 = self.edge_stage1(x)  # 16x480x480

        fisrt = self.edge_stage1_1(stage1) # 16x480x480   
        second = self.edge_stage2_1(self.pool(stage1)) # 16x240x240
        third = self.edge_stage2_2(torch.cat([self.down(fisrt), second],1)) # 32x240x240
        #_ = self.edge_stage1_2(torch.cat([self.up(second), fisrt],1)) # 32x480x480

        fisrt = self.edge_stage2_3(third)   # 32x240x240
        second = self.edge_stage3_1(self.pool(third))  # 32x120x120
        third = self.edge_stage3_2(torch.cat([self.down(fisrt), second],1))  # 64x120x120
        edge_stage2_4 = self.edge_stage2_4(torch.cat([self.up(second), fisrt],1)) # 64x240x240

        fisrt = self.edge_stage3_3(third)   # 64x120x120
        second = self.edge_stage4_1(self.pool(third))  # 64x60x60
        edge_stage4_2 = self.edge_stage4_2(torch.cat([self.down(fisrt), second],1)) #128x60x60
        edge_stage3_4 = self.edge_stage3_4(torch.cat([self.up(second), fisrt],1)) # 128x120x120


        rec_deconc = self.fsp_rgb(edge_stage4_2, stage4_2) ##128x60x60
        rec_edge = self.fsp_hha(stage4_2, edge_stage4_2)  ##128x60x60

        rec_deconc = self.uplayer2(rec_deconc)   ##128x60x60
        rec_edge = self.uplayer2_edge(rec_edge)   ##128x60x60

        rec_edge = self.reduce2_edge(self.up(rec_edge) + edge_stage3_4)  ##    64x120x120
        rec_deconc = self.reduce2(self.up(rec_deconc) + stage3_4)  ## 64x120x120

        rec_deconc = self.fsp_rgb1(rec_edge, rec_deconc) ##64x120x120
        rec_edge = self.fsp_hha1(rec_deconc, rec_edge) ##64x120x120


        rec_deconc = self.uplayer1(rec_deconc)  ##64x120x120
        rec_edge = self.uplayer1_edge(rec_edge) ##64x120x120

        rec_edge = self.reduce1_edge(self.up(rec_edge) + edge_stage2_4)  ##   32x240x240
        rec_deconc = self.reduce1(self.up(rec_deconc) + stage2_4)  ##    32x240x240


        cs = F.interpolate(rec_edge, x_size[2:], mode='bilinear', align_corners=True)        
        cs = self.fuse(cs)
        edge_out = self.sigmoid(cs)

        rec_deconc = F.interpolate(rec_deconc, size=[hei, wid], mode='bilinear')

        diff_edge = self.eirm(edge_out, rec_deconc)
        diff_edge = self.fuse_local(diff_edge)
        diff_edge = self.upsample(diff_edge)
        
        pred = torch.cat([diff_edge, rec_deconc], 1)
        pred = self.resblock(pred)
        pred = self.end(pred)

        return pred, edge_out


    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

if __name__ == '__main__':
    model = EGPNet([4] * 3, [16, 32, 64, 128]).cuda().eval()
    # input = torch.randn(2, 3, 512, 512).cuda()
    # output, edge_out = model(input)
    # print(output.shape)
    # print(edge_out.shape)

    latency = []
    import time
    from thop import profile

    for i in range(100):
        inputs = torch.randn(1, 3, 512, 512).cuda()
        model.eval()

        start_time = time.time()
        outputs = model(inputs)
        latency.append((time.time() - start_time) * 1000)  # 毫秒
    print(f"Latency: {sum(latency) / len(latency):.2f} ms")

    inputs = torch.rand(1, 3, 512, 512).cuda()
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    print(output[-2].shape)

