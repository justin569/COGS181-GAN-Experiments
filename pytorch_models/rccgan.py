import torch
import torch.nn as nn
# from pytorch_models.resnet import Block, ResNet

# usage
# resnet = ResNet(block=ResidualBlock, layers=[3, 4, 6, 3], expansion=1, num_classes=4096)

'''
Basic conditional GAN (cGAN) model
'''

# Discriminator model
# fed in a 3 x 64 x 64 image and label
class ResDiscriminator(nn.Module):
    def __init__(self, params: dict) -> None:
        super(ResDiscriminator, self).__init__()

        self.num_classes = params['num_classes']

        self.embed = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            # in: 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0),
            # out: 1 x 1 x 1
            nn.Flatten()
        )

        self.linear = nn.Linear(self.num_classes+1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        # convert labels to one-hot vectors 
        labels = self.embed(labels)
        # feed image into discriminator
        out = self.model(x)
        # concatenate discriminator output with one-hot vector
        out = out.view(out.shape[0], 1)
        out = torch.cat((out, labels), 1)
        # feed concatenated vector into linear layer
        out = self.linear(out)
        # apply sigmoid to constrain output to [0, 1]
        out = self.sigmoid(out)
        return out

# Generator Model 1 : ResNet-18
class ResGenerator(nn.Module):
    def __init__(self, params: dict) -> None:
        super(ResGenerator, self).__init__()

        self.latent_size = params['latent_size']
        self.num_classes = params['num_classes']
        self.in_channels = 1024
        layers = params['layers']
        resnet_type = params['resnet_type']

        if resnet_type > 34:
            self.expansion = 4
        else:
            self.expansion = 1

        self.emb = nn.Embedding(self.num_classes, self.num_classes)

        self.conv1 = nn.Sequential(
            # in: 138 x 1 x 1
            nn.ConvTranspose2d(self.latent_size+self.num_classes, 1024, 4, 1, 0, bias=False),
            # out: 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # Residual blocks
        self.layer0 = self.make_layers(resnet_type, GenBlock, layers[0], 768, stride=1)
        # out: 768 x 4 x 4
        self.layer1 = self.make_layers(resnet_type, GenBlock, layers[1], 512, stride=2)
        # out: 512 x 8 x 8
        self.layer2 = self.make_layers(resnet_type, GenBlock, layers[2], 256, stride=2)
        # out: 256 x 16 x 16
        self.layer3 = self.make_layers(resnet_type, GenBlock, layers[3], 128, stride=2)
        # out: 128 x 32 x 32
        self.layer4 = self.make_layers(resnet_type, GenBlock, layers[4], 64, stride=2)
        # out: 64 x 64 x 64

        self.conv2 = nn.ConvTranspose2d(64, 3, 5, 1, 2, bias=False)
        # out: 3 x 64 x 64

        self.tanh = nn.Tanh()

    def make_layers(self, resnet_type, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        if stride == 1:
            identity_downsample = nn.Sequential(nn.ConvTranspose2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=3, stride=stride, padding=1),
                                                nn.BatchNorm2d(intermediate_channels*self.expansion))
        else:
            identity_downsample = nn.Sequential(nn.ConvTranspose2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=4, stride=stride, padding=1),
                                                nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(resnet_type, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(resnet_type, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)
    
    def forward(self, x, labels):
        # turn labels into one-hot encoding
        labels = self.emb(labels)
        # reshape to batch_size x num_classes x 1 x 1
        labels = labels.unsqueeze(-1).unsqueeze(-1)
        # concat with latent vector
        x = torch.cat((x, labels), dim=1)

        # feed into model
        out = self.conv1(x)
        # feed into residual layers
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # last conv transpose layer
        out = self.conv2(out)
        # apply tanh function
        out = self.tanh(out)
        return out

class GenBlock(nn.Module):
    def __init__(self, resnet_type, in_channels, out_channels, identity_downsample=None, stride=1):
        assert resnet_type in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(GenBlock, self).__init__()
        self.resnet_type = resnet_type
        if self.resnet_type > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.resnet_type > 34:
            if stride == 1:
                self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            else:
                self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            if stride == 1:
                self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            else:
                self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.resnet_type > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
