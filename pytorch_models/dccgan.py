import torch
import torch.nn as nn


'''
Basic conditional GAN (cGAN) model
'''

# Discriminator model
# fed in a 3 x 64 x 64 image and label
class CondDiscriminator(nn.Module):
    def __init__(self, params: dict) -> None:
        super(CondDiscriminator, self).__init__()

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

# # First Generator Model with optimal arch. from DC-GAN experiments
# class CondGenerator(nn.Module):
#     def __init__(self, params: dict) -> None:
#         super(CondGenerator, self).__init__()

#         self.latent_size = params['latent_size']
#         self.num_classes = params['num_classes']

#         self.emb = nn.Embedding(self.num_classes, self.num_classes)

#         self.model = nn.Sequential(
#             # in: 128 x 1 x 1
#             nn.ConvTranspose2d(self.latent_size+self.num_classes, 1024, 4, 1, 0, bias=False),
#             # out: 1024 x 4 x 4
#             nn.BatchNorm2d(1024),
#             nn.ELU(),
#             nn.ConvTranspose2d(1024, 768, 4, 2, 1, bias=False),
#             # out: 768 x 8 x 8
#             nn.BatchNorm2d(768),
#             nn.ELU(),
#             nn.ConvTranspose2d(768, 512, 4, 2, 1, bias=False),
#             # out: 512 x 16 x 16
#             nn.BatchNorm2d(512),
#             nn.ELU(),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             # out:  256 x 32 x 32
#             nn.BatchNorm2d(256),
#             nn.ELU(),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             # out: 128 x 64 x 64
#             nn.BatchNorm2d(128),
#             nn.ELU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             # out: 64 x 128 x 128
#             nn.BatchNorm2d(64),
#             nn.ELU(),
#             nn.Conv2d(64, 3, 3, 2, 1, bias=False),
#             # out: 3 x 64 x 64
#             nn.Tanh()
#         )
    
#     def forward(self, x, labels):
#         # turn labels into one-hot encoding
#         labels = self.emb(labels)
#         # reshape to batch_size x num_classes x 1 x 1
#         labels = labels.unsqueeze(-1).unsqueeze(-1)
#         # concat with latent vector
#         x = torch.cat((x, labels), dim=1)
#         # feed into model
#         out = self.model(x)
#         return out

# # Generator Model with Avg Pooling
# class CondGenerator(nn.Module):
#     def __init__(self, params: dict) -> None:
#         super(CondGenerator, self).__init__()

#         self.latent_size = params['latent_size']
#         self.num_classes = params['num_classes']

#         self.emb = nn.Embedding(self.num_classes, self.num_classes)

#         self.model = nn.Sequential(
#             # in: 128 x 1 x 1
#             nn.ConvTranspose2d(self.latent_size+self.num_classes, 1024, 4, 1, 0, bias=False),
#             # out: 1024 x 4 x 4
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1024, 768, 4, 2, 1, bias=False),
#             # out: 768 x 8 x 8
#             nn.BatchNorm2d(768),
#             nn.ReLU(),
#             nn.ConvTranspose2d(768, 512, 4, 2, 1, bias=False),
#             # out: 512 x 16 x 16
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             # out:  256 x 32 x 32
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             # out: 128 x 64 x 64
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             # out: 64 x 128 x 128
#             nn.AvgPool2d(5, 2, 2),
#             # out: 64 x 64 x 64
#             nn.Conv2d(64, 3, 3, 1, 1, bias=False),
#             # out: 3 x 64 x 64
#             nn.Tanh()
#         )
    
#     def forward(self, x, labels):
#         # turn labels into one-hot encoding
#         labels = self.emb(labels)
#         # reshape to batch_size x num_classes x 1 x 1
#         labels = labels.unsqueeze(-1).unsqueeze(-1)
#         # concat with latent vector
#         x = torch.cat((x, labels), dim=1)
#         # feed into model
#         out = self.model(x)
#         return out

# Generator Model with 2x more neurons per layer
# Generator Model with 2x more neurons per layer and dropout
class CondGenerator(nn.Module):
    def __init__(self, params: dict) -> None:
        super(CondGenerator, self).__init__()

        self.latent_size = params['latent_size']
        self.num_classes = params['num_classes']

        self.emb = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            # in: 138 x 1 x 1
            nn.ConvTranspose2d(self.latent_size+self.num_classes, 2048, 4, 1, 0, bias=False),
            # out: 2048 x 4 x 4
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            # out: 1024 x 8 x 8
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            # out: 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            # out:  256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            # out: 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            # out: 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.Conv2d(64, 3, 3, 2, 1, bias=False),
            # out: 3 x 64 x 64
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        # turn labels into one-hot encoding
        labels = self.emb(labels)
        # reshape to batch_size x num_classes x 1 x 1
        labels = labels.unsqueeze(-1).unsqueeze(-1)
        # concat with latent vector
        x = torch.cat((x, labels), dim=1)
        # feed into model
        out = self.model(x)
        return out