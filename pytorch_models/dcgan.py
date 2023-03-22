import torch
import torch.nn as nn


'''
Basic GAN model
'''

# Discriminator model
# fed in a 3 x 64 x 64 image
class BaseDiscriminator(nn.Module):
    def __init__(self, params: dict) -> None:
        super(BaseDiscriminator, self).__init__()
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
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        out = self.model(x)
        return out

# Base Generator Model: same architecture as specified paper
# class BaseGenerator(nn.Module):
#     def __init__(self, params: dict) -> None:
#         super(BaseGenerator, self).__init__()

#         self.latent_size = params['latent_size']

#         self.model = nn.Sequential(
#             # in: 128 x 1 x 1
#             nn.ConvTranspose2d(self.latent_size, 1024, 4, 1, 0, bias=False),
#             # out: 1024 x 4 x 4
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
#             # out: 512 x 8 x 8
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             # out: 256 x 16 x 16
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             # out:  64 x 32 x 32
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
#             # out: 3 x 64 x 64
#             nn.Tanh()
#         )
    
#     def forward(self, x, labels):
#         out = self.model(x)
#         return out

# Base Generator Model: double the neurons in each layer
# class BaseGenerator(nn.Module):
#     def __init__(self, params: dict) -> None:
#         super(BaseGenerator, self).__init__()

#         self.latent_size = params['latent_size']

#         self.model = nn.Sequential(
#             # in: 128 x 1 x 1
#             nn.ConvTranspose2d(self.latent_size, 2048, 4, 1, 0, bias=False),
#             # out: 1024 x 4 x 4
#             nn.BatchNorm2d(2048),
#             nn.ReLU(),
#             nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
#             # out: 512 x 8 x 8
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
#             # out: 256 x 16 x 16
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             # out:  64 x 32 x 32
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 3, 4, 2, 1, bias=False),
#             # out: 3 x 64 x 64
#             nn.Tanh()
#         )
    
#     def forward(self, x, labels):
#         out = self.model(x)
#         return out

# Base Generator Model: double the neurons in each layer, 2 more layers
# class BaseGenerator(nn.Module):
#     def __init__(self, params: dict) -> None:
#         super(BaseGenerator, self).__init__()

#         self.latent_size = params['latent_size']

#         self.model = nn.Sequential(
#             # in: 128 x 1 x 1
#             nn.ConvTranspose2d(self.latent_size, 2048, 4, 1, 0, bias=False),
#             # out: 1024 x 4 x 4
#             nn.BatchNorm2d(2048),
#             nn.ReLU(),
#             nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
#             # out: 512 x 8 x 8
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
#             # out: 256 x 16 x 16
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             # out:  64 x 32 x 32
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             # out: 128 x 64 x 64
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             # out: 64 x 128 x 128
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 3, 3, 2, 1, bias=False),
#             # out: 3 x 64 x 64
#             nn.Tanh()
#         )
    
#     def forward(self, x, labels):
#         out = self.model(x)
#         return out

# Base Generator Model: more neurons in second conv layer, 2 more layers
class BaseGenerator(nn.Module):
    def __init__(self, params: dict) -> None:
        super(BaseGenerator, self).__init__()

        self.latent_size = params['latent_size']

        self.model = nn.Sequential(
            # in: 128 x 1 x 1
            nn.ConvTranspose2d(self.latent_size, 1024, 4, 1, 0, bias=False),
            # out: 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 768, 4, 2, 1, bias=False),
            # out: 768 x 8 x 8
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.ConvTranspose2d(768, 512, 4, 2, 1, bias=False),
            # out: 256 x 16 x 16
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            # out:  64 x 32 x 32
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            # out: 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            # out: 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, 2, 1, bias=False),
            # out: 3 x 64 x 64
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        out = self.model(x)
        return out
