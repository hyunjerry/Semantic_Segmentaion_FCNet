import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    #### This - check the architecture!
    def forward(self, x):
        # Encoder
        # print('encoder')
        x = self.features_block1(x)
        # print(f'pool1 {x.shape}')
        x = self.features_block2(x)
        # print(f'pool2 {x.shape}')
        x = self.features_block3(x)
        # print(f'pool3 {x.shape}')
        pool3_score = self.score_pool3(x)
        # print(f'pool3_score {pool3_score.shape}')
        x = self.features_block4(x)
        # print(f'pool4 {x.shape}')
        pool4_score = self.score_pool4(x)
        # print(f'pool4_score {pool4_score.shape}')
        x = self.features_block5(x)
        # print(f'pool5 {x.shape}')
        
        # Classifier
        x = self.classifier(x)
        # print(x.shape)
        
        # Decoder
        # print('decoder')
        x = self.upscore2(x)
        x = x[:, :, 1:-1, 1:-1]
        # print(x.shape)

        # First fusion
        x = x + pool4_score
        # print('First fusion')
        x = self.upscore2(x)
        x = x[:, :, 1:-1, 1:-1]
        # print(x.shape)

        # Second fusion
        # print('Second fusion')
        x = x + pool3_score
        # print(x.shape)
        x = self.upscore_final(x)
        x = x[:, :, 4:-4, 4:-4]
        # print(x.shape)
        
        return x
    