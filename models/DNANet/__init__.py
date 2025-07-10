from .model_DNANet import DNANet, Res_CBAM_block

def get_DNANet():
 return DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[1, 1, 1, 1],
                   nb_filter=[16, 32, 64, 128, 256], deep_supervision=False).cuda()