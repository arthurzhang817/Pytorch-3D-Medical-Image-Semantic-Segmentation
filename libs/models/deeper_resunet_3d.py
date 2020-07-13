import torch
import torch.nn as nn
import torch.nn.functional as F

def deeper_resunet_3d(n_classes, base_filters, channel_in):
    model_in_block = in_block(channel_in=channel_in, channel_out=base_filters)
    model_encoder = encoder(base_filters=base_filters)
    model_decoder = decoder(base_filters=base_filters)
    model_seg_out_block = seg_out_block(base_filters=base_filters, n_classes=n_classes)

    model = seg_path(
        model_in_block,
        model_encoder,
        model_decoder,
        model_seg_out_block
    )

    return model

class in_block(nn.Module):
    '''
    in_block is used to connect the input of the whole network.

    number of channels is changed by conv1, and then it keeps the same for all
    following layers.

    parameters:
        channel_in: int
            the number of channels of the input.
            RGB images have 3, greyscale images have 1, etc.
        channel_out: int
            the number of filters for conv1; keeps unchanged for all layers following
            conv1

    '''
    def __init__(self, channel_in, channel_out):
        super(in_block, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = nn.Conv3d(
            kernel_size=3,
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            padding=1
        )
        self.bn1 = nn.BatchNorm3d(num_features=self.channel_out)

        self.conv2 = nn.Conv3d(
            kernel_size=3,
            in_channels=self.channel_out,
            out_channels=self.channel_out,
            padding=1
        )

        self.conv3 = nn.Conv3d(
            kernel_size=3,
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            padding=1
        )
        self.bn3 = nn.BatchNorm3d(num_features=self.channel_out)

    def forward(self, x):
        path = self.conv1(x)
        path = self.bn1(path)
        path = F.leaky_relu(path)
        path = F.dropout(path, p=0.2)

        path = self.conv2(path)

        residual = self.conv3(x)
        residual = self.bn3(residual)

        self.down_level1 = path + residual

        return self.down_level1

class res_block(nn.Module):
    '''
    res_block used for down and up, toggled by downsample.

    "input" -> bn1 -> relu1 -> conv1 -> bn2 -> relu2 -> conv2 -> "path"
            -> conv3 -> bn3 -> "residual"
    
    return "output" = "path" + "residual"

    downsampling (if any) is done by conv1

    parameters:
        channel_in: int
        downsample: boolean 
            if downsample is true, the block is used for encoding path,
            during which the channels out are doubled by the conv1. 
            conv1 will have stride 2.

            if downsample is false, the block is used for segmenting/restoring 
            path, during which the channels keep the same through the block. 
            conv1 will have stride 1.

    '''
    def __init__(
        self,
        channel_in,
        downsample=False,
    ):
        super(res_block, self).__init__()

        self.channel_in = channel_in

        if downsample:
            self.channel_out = 2*self.channel_in
            self.conv1_stride = 2
            self.conv3_stride = 2
        else:
            self.channel_out = self.channel_in
            self.conv1_stride = 1
            self.conv3_stride = 1
        
        self.bn1 = nn.BatchNorm3d(num_features=self.channel_in)
        self.conv1 = nn.Conv3d(
                in_channels=self.channel_in,
                kernel_size=3,
                out_channels=self.channel_out,
                stride=self.conv1_stride,
                padding=1
                )
        self.bn2 = nn.BatchNorm3d(num_features=self.channel_out)
        self.conv2 = nn.Conv3d(
            in_channels=self.channel_out,
            out_channels=self.channel_out,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv3d(
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            stride=self.conv3_stride,
            padding=1,
            kernel_size=3
        )
        self.bn3 = nn.BatchNorm3d(num_features=self.channel_out)

    def forward(self, x):

        path = self.bn1(x)
        path = F.leaky_relu(path)
        path = F.dropout(path, p=0.2)

        path = self.conv1(path)
        path = self.bn2(path)
        path = F.leaky_relu(path)
        path = F.dropout(path, p=0.2)

        path = self.conv2(path)

        residual = self.conv3(x)
        residual = self.bn3(residual)

        output = path + residual

        return output

class encoder(nn.Module):

    '''
    encoder

    dataflow:
    x --down_block2--> down_level2 
    --down_block3--> down_level3 
    --down_block4--> down_level4 
    --down_bridge--> codes

    parameters:
        base_filters: number of filters received from in_block; 16 by default.

    '''
    def __init__(
        self,
        base_filters
    ):
        super(encoder, self).__init__()

        self.bf = base_filters

        self.down_block2 = res_block(
            channel_in=self.bf ,
            downsample=True
        )
        self.down_block3 = res_block(
            channel_in=self.bf *2,
            downsample=True
        )
        self.down_block4 = res_block(
            channel_in=self.bf *4,
            downsample=True
        )
        self.down_bridge = res_block(
            channel_in=self.bf *8,
            downsample=True
        )

    def forward(self, x):

        self.down_level2 = self.down_block2(x)
        self.down_level3 = self.down_block3(self.down_level2)
        self.down_level4 = self.down_block4(self.down_level3)
        self.codes = self.down_bridge(self.down_level4)

        return self.codes

class decoder(nn.Module):
    '''
    decoder

    dataflow:
    x --upsample4--> up4 --up_block4--> up_level4 
    --upsample3--> up3 --up_block3--> up_level3
    --upsample2--> up2 --up_block2--> up_level2
    --upsample1--> up1 --up_block1--> up_level1

    parameters:
        base_filters: number of filters consistent with encoder; 16 by default.

    '''
    def __init__(
        self,
        base_filters
    ):
        super(decoder, self).__init__()
        self.bf = base_filters
        
        self.upsample4 = nn.ConvTranspose3d(
            in_channels=self.bf*16 ,
            out_channels=self.bf*8 , 
            kernel_size=2, 
            stride=2
        )
        self.conv4 = nn.Conv3d(
            in_channels=self.bf*16,
            out_channels=self.bf*8,
            kernel_size=1
        )
        self.up_block4 = res_block(
            channel_in=self.bf*8,
            downsample=False
        )

        self.upsample3 = nn.ConvTranspose3d(
            in_channels=self.bf*8,
            out_channels=self.bf*4,
            kernel_size=2,
            stride=2
        )
        self.conv3 = nn.Conv3d(
            in_channels=self.bf*8,
            out_channels=self.bf*4,
            kernel_size=1
        )
        self.up_block3 = res_block(
            channel_in=self.bf*4,
            downsample=False
        )

        self.upsample2 = nn.ConvTranspose3d(
            in_channels=self.bf*4,
            out_channels=self.bf*2,
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv3d(
            in_channels=self.bf*4,
            out_channels=self.bf*2,
            kernel_size=1
        )
        self.up_block2 = res_block(
            channel_in=self.bf*2,
            downsample=False
        )

        self.upsample1 = nn.ConvTranspose3d(
            in_channels=self.bf*2,
            out_channels=self.bf,
            kernel_size=2,
            stride=2
        )
        self.conv1 = nn.Conv3d(
            in_channels=self.bf*2,
            out_channels=self.bf,
            kernel_size=1
        )
        self.up_block1 = res_block(
            channel_in=self.bf,
            downsample=False
        )
    
    def forward(self, x):
        
        up4 = self.upsample4(x)
        self.up_level4 = self.up_block4(up4)

        up3 = self.upsample3(self.up_level4)
        self.up_level3 = self.up_block3(up3)

        up2 = self.upsample2(self.up_level3)
        self.up_level2 = self.up_block2(up2)

        up1 = self.upsample1(self.up_level2)
        self.up_level1 = self.up_block1(up1)

        return self.up_level1

class seg_out_block(nn.Module):
    '''
    seg_out_block, receive data from decoder and output the segmentation mask

    parameters:
        base_filters: number of filters received from in_block.
        n_classes: number of classes

    '''
    def __init__(
        self,
        base_filters,
        n_classes=6
    ):
        super(seg_out_block, self).__init__()

        self.bf = base_filters
        self.n_classes = n_classes
        self.conv = nn.Conv3d(
            in_channels=self.bf,
            out_channels=self.n_classes,
            kernel_size=1
        )
    
    def forward(self, x):
        self.output = self.conv(x)
        return self.output

class seg_path(nn.Module):
    def __init__(
        self,
        in_block,
        encoder,
        decoder,
        seg_out_block
    ):
        super(seg_path, self).__init__()

        self.in_block = in_block
        self.encoder = encoder
        self.decoder = decoder
        self.seg_out_block = seg_out_block

    def forward(self, x):

        self.down_level1 = self.in_block(x)

        self.down_level2 = self.encoder.down_block2(self.down_level1)
        self.down_level3 = self.encoder.down_block3(self.down_level2)
        self.down_level4 = self.encoder.down_block4(self.down_level3)
        self.codes = self.encoder.down_bridge(self.down_level4)

        self.up4 = self.decoder.upsample4(self.codes)
        up4_dummy = torch.cat([self.up4, self.down_level4],1)
        up4_dummy = self.decoder.conv4(up4_dummy)
        self.up_level4 = self.decoder.up_block4(up4_dummy)

        self.up3 = self.decoder.upsample3(self.up_level4)
        up3_dummy = torch.cat([self.up3, self.down_level3], 1)
        up3_dummy = self.decoder.conv3(up3_dummy)
        self.up_level3 = self.decoder.up_block3(up3_dummy)

        self.up2 = self.decoder.upsample2(self.up_level3)
        up2_dummy = torch.cat([self.up2, self.down_level2], 1)
        up2_dummy = self.decoder.conv2(up2_dummy)
        self.up_level2 = self.decoder.up_block2(up2_dummy)

        self.up1 = self.decoder.upsample1(self.up_level2)
        up1_dummy = torch.cat([self.up1, self.down_level1], 1)
        up1_dummy = self.decoder.conv1(up1_dummy)
        self.up_level1 = self.decoder.up_block1(up1_dummy)

        self.output = self.seg_out_block(self.up_level1)

        return self.output