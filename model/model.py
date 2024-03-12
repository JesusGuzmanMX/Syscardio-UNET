'''
Aquí se encuentra la arquitectura del modelo UNet utilizado

'''


from torch import nn 
import torch


class UNet(nn.Module):

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                    # 2 convoluciones de 3x3
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,
                                    stride = 1, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                                    stride = 1, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                )
        return block

    def expansive_block(self, in_channels, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,
                                    stride = 1, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                                    stride = 1, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels)
                    )
            return  block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,
                                    stride = 1, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=1, in_channels=mid_channel, out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels)
                    )
            return  block


    #Bottleneck
    def bottleneck(self, in_channels, out_channels, kernel_size = 3):
            block  = torch.nn.Sequential(
                            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,
                                            stride = 1, padding = 1),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                                            stride = 1,padding = 1),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(out_channels)
                            )
            return block

    def __init__(self):
        super(UNet, self).__init__()
        #Encode
        #[N,1,128,128]=>[N,16,128,128]=>[N,16,128,128]
        self.conv_encode1 = self.contracting_block(in_channels=1, out_channels=16)
        #[N,16,128,128] => [N,16,64,64]
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        #[N,16,64,64]=>[N,32,64,64]=>[N,32,64,64]
        self.conv_encode2 = self.contracting_block(16, 32)
        #[N,32,64,64] => [N,32,32,32]
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        #[N,32,32,32]=>[N,64,32,32]=>[N,64,32,32]
        self.conv_encode3 = self.contracting_block(32, 64)
        #[N,64,32,32] => [N,64,16,16]
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        #[N,64,16,16]=>[N,128,16,16]=>[N,128,16,16]
        self.conv_encode4 = self.contracting_block(64, 128)
        #[N,128,16,16] => [N,128,8,8]
        self.conv_maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        # bottleneck
        self.bn = self.bottleneck(128,256)

        # Decode
        #[N,256,8,8] => [N,256,16,16]
        self.conv_up4 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256,kernel_size=2, stride=2)
        #([N,256,16,16][N,128,16,16]) => [N,128,16,16]
        # salida de conv_up + encode y sale del tamaño de encode
        self.conv_decode4 = self.expansive_block(384,128)

        #[N,128,16,16] => [N,128,32,32]
        self.conv_up3 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128,kernel_size=2, stride=2)
        #([N,128,32,32][N,64,32,32]) => [N,64,32,32]
        self.conv_decode3 = self.expansive_block(192,64)

        # [N,64,32,32] => [N,64,64,64]
        self.conv_up2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64,kernel_size=2, stride=2)
        # ([N,64,64,64][N,32,64,64]) => [N,32,64,64]
        self.conv_decode2 = self.expansive_block(96,32)

        # [N,32,64,64] => [N,32,128,128]
        self.conv_up1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32,
                                                 kernel_size=2, stride=2)
        # ([N,32,128,128][N,16,128,128]) => [N,16,128,128]
        self.conv_decode1 = self.expansive_block(48,16)

        # Tail
        #[N,16,128,128]=>[N,8,128,128] =>[N,5,128,128]
        self.final = self.final_block(16,8,5)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode

        # [N,1,128,128] => [N,16,128,128] # concat this
        encode_block1 = self.conv_encode1(x)
        # [N,16,128,128] => [N,16,64,64]
        encode_pool1 = self.conv_maxpool1(encode_block1)

        # [N,16,16,64] => [N,32,64,64] # concat this
        encode_block2 = self.conv_encode2(encode_pool1)
        # [N,32,64,64] => [N,32,32,32]
        encode_pool2 = self.conv_maxpool2(encode_block2)

        # [N,32,32,32] => [N,64,32,32] # concat this
        encode_block3 = self.conv_encode3(encode_pool2)
        # [N,64,32,32] => [N,64,16,16]
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # [N,64,16,16] => [N,128,16,16] # concat this
        encode_block4 = self.conv_encode4(encode_pool3)
        # [N,128,16,16] => [N,128,8,8]
        encode_pool4 = self.conv_maxpool4(encode_block4)

        # Bottleneck
        # [N,128,8,8] => [N,256,8,8]
        bottleneck1 = self.bn(encode_pool4)

        # Decode
        #[N,256,8,8] => [N,256,16,16]
        decode_up4 = self.conv_up4(bottleneck1)
        # ([N,256,16,16][N,128,16,16])
        cat_layer4 = self.crop_and_concat(decode_up4,encode_block4,crop = False)
        # [N,384,16,16] => [N,128,16,16]
        decode_block4 = self.conv_decode4(cat_layer4)


        #[N,128,16,16] => [N,128,32,32]
        decode_up3 = self.conv_up3(decode_block4)
        # ([N,128,32,32][N,64,32,32])
        cat_layer3 = self.crop_and_concat(decode_up3,encode_block3,crop = False)
        # [N,192,32,32] => [N,64,32,32]
        decode_block3 = self.conv_decode3(cat_layer3)

        #[N,64,32,32] => [N,64,64,64]
        decode_up2 = self.conv_up2(decode_block3)
        # ([N,64,64,64][N,32,64,64])
        cat_layer2 = self.crop_and_concat(decode_up2,encode_block2, crop = False)
        # [N,96,64,64] => [N,32,64,64]multiclass
        decode_block2 = self.conv_decode2(cat_layer2)

        #[N,32,64,64] => [N,32,128,128]
        decode_up1 = self.conv_up1(decode_block2)
        # ([N,32,128,128][N,16,128,128])
        cat_layer1 = self.crop_and_concat(decode_up1,encode_block1, crop = False)
        # [N,48,128,128] => [N,16,128,128]
        decode_block1 = self.conv_decode1(cat_layer1)

        # [N,16,128,128] => [N,5,128,128]
        final_layer = self.final(decode_block1)

        return  final_layer