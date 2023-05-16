import torch
import torch.nn as nn

'''
This model is fitted for training images with the size 512x512
'''

'''
Class to create the convolutional block
'''
class conv_block(nn.Module):
    '''
    Init function to initialize the convolutional block
    INPUT:
        in_c : number of input channels
        out_c : number of output channels
    '''
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
    '''
    Forward function to perform the forward pass
    INPUT:
        inputs : input to the forward pass
    OUTPUT:
        x : output of the forward pass
    '''
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

'''
Class to create the encoder block
'''
class encoder_block(nn.Module):
    '''
    Init function to initialize the encoder block
    INPUT:
        in_c : number of input channels
        out_c : number of output channels
    '''
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    '''
    Forward function to perform the forward pass
    INPUT:
        inputs : input to the forward pass
    OUTPUT:
        x : output of the forward pass
        p : output of the pooling layer
    '''
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

'''
Class to create the decoder block
'''
class decoder_block(nn.Module):
    '''
    Init function to initialize the decoder block
    INPUT:
        in_c : number of input channels
        out_c : number of output channels
    '''
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    '''
    Forward function to perform the forward pass
    INPUT:
        inputs : input to the forward pass
        skip : skip connection from the encoder block
    OUTPUT:
        x : output of the forward pass
    '''
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x) 
        return x

'''
Class to create the UNet model
'''
class build_unet(nn.Module):
    '''
    Init function to initialize the UNet model
    '''
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    '''
    Forward function to perform the forward pass
    INPUT:
        inputs : input to the forward pass
    OUTPUT: 
        outputs : output of the forward pass
    '''
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape)