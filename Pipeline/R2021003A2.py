import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import time
from torchvision import transforms
import torch.nn.functional as F
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import time
from sklearn.model_selection import train_test_split
import math



"""
Write Code for Downloading Image and Audio Dataset Here
"""


####-----------------------------------------------------Image Dataset--------------------------------------------------------------####


### Image Dataset Class
class ImageDataset(Dataset):
    def __init__(self, split: str = "train", root: str = "./data") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if split in ["train", "val"]:
            full_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=self.transform)
            train_idx, val_idx = train_test_split(
                np.arange(len(full_dataset)),
                test_size=0.2,
                random_state=42,
                shuffle=True
            )
            
            if split == "train":
                self.indices = train_idx
            else:
                self.indices = val_idx
            
            self.dataset = torch.utils.data.Subset(full_dataset, self.indices)
        elif split == "test":
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=self.transform)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.datasplit in ["train", "val", "test"]:
            # Since the dataset is a Subset, idx is already mapped to the correct index in the subset
            image, label = self.dataset[idx]
            return image, label
        else:
            raise Exception("Data split must be in [train, test, val]")        

### Function to download Audio Dataset
def download_speechcommands(root: str = "./data", subset: str = None):
    return torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=True)

####-----------------------------------------------------Audio Dataset--------------------------------------------------------------####


### Audio Dataset Class
class AudioDataset(Dataset):
    def __init__(self, split: str = "train", root: str = "./data") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        if split == "train":
            self.dataset = download_speechcommands(root, subset="training")
        elif split == "test":
            self.dataset = download_speechcommands(root, subset="testing")
        else:
            # Assuming you have a mechanism to create a validation set
            self.dataset = download_speechcommands(root, subset="validation")

        # Define a resampler to resample audio to 1600 Hz
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=1600)  # Adjusted for 1600 Hz
        self.label_to_index = {"backward":0, "bed":1, "bird":2, "cat":3, "dog":4, "down":5, "eight":6, "five":7,"follow":8, "forward":9, "four":10, "go":11, "happy":12, "house":13, "learn":14,"left":15, "marvin":16, "nine":17, "no":18, "off":19, "on":20, "one":21, "right":22,"seven":23, "sheila":24, "six":25, "stop":26, "three":27, "tree":28, "two":29, "up":30,"visual":31, "wow":32, "yes":33,"zero":34}


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, _, _ = self.dataset[idx]
        
        # Resample if necessary
        if sample_rate != 1600:
            waveform = self.resampler(waveform)

        # Perform padding if necessary to match the maximum length in the dataset
        max_length = 1600   # Example for a maximum length of 10 seconds at 1600 Hz
        if waveform.size(1) < max_length:
            # Padding
            padded_waveform = torch.zeros((waveform.size(0), max_length))
            padded_waveform[:, :waveform.size(1)] = waveform
        else:
            padded_waveform = waveform[: , :max_length]  # Truncate if longer than max_length
        
        return padded_waveform, self.label_to_index[label]
    
####-----------------------------------------------------RESNET IMAGE--------------------------------------------------------------####


class ResidualBlockImage(nn.Module):
    def __init__(self, in_channels, use_1x1conv=False):
        super(ResidualBlockImage, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        if use_1x1conv: # Define the 1x1 conv layer for 
            self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        else:
            self.conv3 = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            identity = self.conv3(identity)
        out += identity
        return self.relu(out)
    
####-----------------------------------------------------RESNET AUDIO--------------------------------------------------------------####


class ResidualBlockAudio(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlockAudio, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

####-----------------------------------------------------RESNET--------------------------------------------------------------####

class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        ### IMAGE
        self.num_classes_image = 10
        self.in_channels_image = 16 # self.in_channels is 64 because 
        self.conv_image = nn.Conv2d(3, self.in_channels_image, kernel_size=3, padding=1)
        self.bn_image = nn.BatchNorm2d(self.in_channels_image)
        self.relu_image = nn.ReLU(inplace=True)
        self.resblocks_image = self._make_layer(18)  # 18 residual blocks
        self.avgpool_image = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_image = nn.Linear(self.in_channels_image, self.num_classes_image)


        ### AUDIO
        self.num_classes_audio = 35
        self.in_channels_audio = 1
        self.initial_conv_audio = nn.Conv1d(self.in_channels_audio, 16, kernel_size=3, padding=1, bias=False)
        self.initial_bn_audio = nn.BatchNorm1d(16)
        self.initial_relu_audio = nn.ReLU()
        self.res_blocks_audio = nn.Sequential(*[ResidualBlockAudio(16) for _ in range(18)])
        self.final_pool_audio = nn.AdaptiveAvgPool1d(1)
        self.fc_audio = nn.Linear(16, self.num_classes_audio)
                

    def _make_layer(self, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlockImage(self.in_channels_image))
        return nn.Sequential(*layers)

    def forward(self, x):
        if(len(x.shape) == 4):
            x = self.relu_image(self.bn_image(self.conv_image(x)))
            x = self.resblocks_image(x)
            x = self.avgpool_image(x)
            x = torch.flatten(x, 1)
            x = self.fc_image(x)
        else:
            x = self.initial_relu_audio(self.initial_bn_audio(self.initial_conv_audio(x)))
            x = self.res_blocks_audio(x)
            x = self.final_pool_audio(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.fc_audio(x)

        return x
    


####-----------------------------------------------------VGG--------------------------------------------------------------####
        
def next_size(size):
    # Increase by 25% and round up to the nearest odd integer
    return math.ceil(size * 1.25) | 1  # The bitwise OR operation with 1 ensures the number is odd

def next_channels(channels):
    # Reduce by 35% and use the ceil function
    return math.ceil(channels * 0.65)

####-----------------------------------------------------VGG IMAGE--------------------------------------------------------------####

def vgg_block_image(num_convs, in_channels, out_channels, kernel_size=3):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels  # Set for next conv layer within the same block
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Add max pooling layer
    return nn.Sequential(*layers)

####-----------------------------------------------------VGG AUDIO--------------------------------------------------------------####


def vgg_block_audio(num_convs, in_channels, out_channels, kernel_size=3):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels  # Set for the next conv layer within the same block
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # Add max pooling layer for 1D data
    return nn.Sequential(*layers)


####-----------------------------------------------------VGG--------------------------------------------------------------####

class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        current_channels_image = 3
        
        current_kernel_size = 3
        ### IMAGE
        self.num_classes_image = 10  # CIFAR-10 has 10 classes

        # Block 1
        self.block1_image = vgg_block_image(2, current_channels_image, 32, current_kernel_size)
        out_channels = next_channels(32)  # Compute the next number of channels
        kernel_size = next_size(current_kernel_size)  # Compute the next kernel size

        # Block 2
        self.block2_image = vgg_block_image(2, 32, out_channels, kernel_size)
        current_channels = out_channels  # Update current_channels to the output of the last block
        out_channels = next_channels(out_channels)  # Compute the next number of channels
        kernel_size = next_size(kernel_size)  # Compute the next kernel size

        # Block 3
        self.block3_image = vgg_block_image(3, current_channels, out_channels, kernel_size)
        current_channels = out_channels  # Update current_channels to the output of the last block
        out_channels = next_channels(out_channels)  # Compute the next number of channels
        kernel_size = next_size(kernel_size)  # Compute the next kernel size

        # Block 4
        self.block4_image = vgg_block_image(3, current_channels, out_channels, kernel_size)
        current_channels = out_channels  # Update current_channels to the output of the last block
        out_channels = next_channels(out_channels)  # Compute the next number of channels
        kernel_size = next_size(kernel_size)  # Compute the next kernel size

        # Block 5
        self.block5_image = vgg_block_image(3, current_channels, out_channels, kernel_size)

        # Classifier remains unchanged
        self.classifier_image = nn.Sequential(
            nn.Linear(out_channels * 1 * 1, 256), 
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(128, self.num_classes_image)
        )

        ### AUDIO

        current_channels_audio = 1
        self.num_classes_audio = 35  # SpeechCommands has 35 classes

        self.block1_audio = vgg_block_audio(2, current_channels_audio, 32, current_kernel_size)
        out_channels = next_channels(32)
        kernel_size = next_size(current_kernel_size)

        self.block2_audio = vgg_block_audio(2, 32, out_channels, kernel_size)
        current_channels = out_channels
        out_channels = next_channels(out_channels)
        kernel_size = next_size(kernel_size)

        self.block3_audio = vgg_block_audio(3, current_channels, out_channels, kernel_size)
        current_channels = out_channels
        out_channels = next_channels(out_channels)
        kernel_size = next_size(kernel_size)

        self.block4_audio = vgg_block_audio(3, current_channels, out_channels, kernel_size)
        current_channels = out_channels
        out_channels = next_channels(out_channels)
        kernel_size = next_size(kernel_size)

        self.block5_audio = vgg_block_audio(3, current_channels, out_channels, kernel_size)


        self.classifier_audio = nn.Sequential(
            nn.Linear(350 * 1 * 1, 256), 
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes_audio)
        )

        self._initialize_weights()

    def forward(self, x):
        if(len(x.shape) == 4):
            x = self.block1_image(x)
            x = self.block2_image(x)
            x = self.block3_image(x)
            x = self.block4_image(x)
            x = self.block5_image(x)
            x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
            x = self.classifier_image(x)
        else:
            x = self.block1_audio(x)
            x = self.block2_audio(x)
            x = self.block3_audio(x)
            x = self.block4_audio(x)
            x = self.block5_audio(x)
            x = torch.flatten(x, 1)
            x = self.classifier_audio(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

####-----------------------------------------------------Inception Module Image--------------------------------------------------------------####


class InceptionModuleImage(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModuleImage, self).__init__()
        # Reduced number of channels in each branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),  # Reduced from 64 to 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.branch5x5_1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1),  # Reduced from 48 to 24
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.branch5x5_2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=5, padding=2),  # Output also reduced to 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.branch3x3dbl_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),  # Adjusted to match branch1x1 output
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch3x3dbl_2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),  # Reduced from 96 to 48
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.branch3x3dbl_3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),  # Kept consistent with branch3x3dbl_2 output
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 16, kernel_size=1),  # Reduced from 32 to 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        # Concatenate along the channel dimension
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
    
####-----------------------------------------------------Inception Module Audio--------------------------------------------------------------####


class AudioInceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(AudioInceptionModule, self).__init__()
        # Significantly increase the number of channels in each branch
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=1),  # Significantly increased
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.branch5x5_1 = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=1),  # Significantly increased
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.branch5x5_2 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=5, padding=2),  # Aligned with branch1x1 increase
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.branch3x3dbl_1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=1),  # Significantly increased
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.branch3x3dbl_2 = nn.Sequential(
            nn.Conv1d(128, 192, kernel_size=3, padding=1),  # Significantly increased
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )
        self.branch3x3dbl_3 = nn.Sequential(
            nn.Conv1d(192, 192, kernel_size=3, padding=1),  # Maintained to match branch3x3dbl_2
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, 64, kernel_size=1),  # Significantly increased
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = self.branch_pool(x)

        outputs = torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)
        return outputs



        
class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

        self.num_classes_image = 10

        self.inception1_image = InceptionModuleImage(in_channels=3)  # CIFAR-10 images have 3 channels
        
        # Output channels from inception1 is the sum of all output channels from its reduced branches
        inception1_out_channels_image = 32 + 32 + 48 + 16  # Updated to reflect reduced channels
        
        self.inception2_image = InceptionModuleImage(in_channels=inception1_out_channels_image)
        # Output channels remain consistent for each subsequent inception block based on the reduced design
        inception2_out_channels_image = 32 + 32 + 48 + 16  # Keep consistent with reduced channel design
        
        self.inception3_image = InceptionModuleImage(in_channels=inception2_out_channels_image)
        inception3_out_channels_image = 32 + 32 + 48 + 16  # Consistent with above
        
        self.inception4_image = InceptionModuleImage(in_channels=inception3_out_channels_image)
        inception4_out_channels_image = 32 + 32 + 48 + 16  # Consistent with above
        
        # Assuming the size of the image entering the classifier is adjusted for reduced output channels
        self.avgpool_image = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_image = nn.Linear(inception4_out_channels_image, self.num_classes_image)  # Adjusted for reduced channels


        self.num_classes_audio = 35

        self.inception1_audio = AudioInceptionModule(in_channels=1)
        inception1_out_channels_audio = 128 + 128 + 192 + 64  # Adjusted according to further reduced channel sizes
        
        self.inception2_audio = AudioInceptionModule(in_channels=inception1_out_channels_audio)
        inception2_out_channels_audio = 128 + 128 + 192 + 64
        
        self.inception3_audio = AudioInceptionModule(in_channels=inception2_out_channels_audio)
        inception3_out_channels_audio = 128 + 128 + 192 + 64
        
        self.inception4_audio = AudioInceptionModule(in_channels=inception3_out_channels_audio)
        inception4_out_channels_audio = 128 + 128 + 192 + 64 
        
        self.avgpool_audio = nn.AdaptiveAvgPool1d(1)
        self.classifier_audio = nn.Linear(inception4_out_channels_audio, self.num_classes_audio)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        if(len(x.shape)==4):
            x = self.inception1_image(x)
            x = self.inception2_image(x)
            x = self.inception3_image(x)
            x = self.inception4_image(x)
            x = self.avgpool_image(x)
            x = torch.flatten(x, 1)
            x = self.classifier_image(x)
        else:
            x = self.inception1_audio(x)
            x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
            x = self.inception2_audio(x)
            ### Apply max pooling to reduce the size of the output
            x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
            x = self.inception3_audio(x)
            x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
            x = self.inception4_audio(x)
            x = self.avgpool_audio(x)
            x = torch.flatten(x, 1)
            x = self.classifier_audio(x)
        return x
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)



class ResidualBlock_CustomImage(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResidualBlock_CustomImage, self).__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 is not None:
            identity = self.conv3(identity)
        out += identity
        return self.relu(out)
    
class InceptionModule_CustomImage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule_CustomImage, self).__init__()
        
        # Define the number of output channels for each branch based on the desired total output channels
        # Here, we're simply dividing the output channels evenly across branches for simplicity
        # Adjust these proportions as needed for your specific architecture
        rem = out_channels % 4
        branch_channels = out_channels // 4
        branch_channels1 = branch_channels
        if(rem!=0):
            branch_channels1 += 1
            rem-=1
        branch_channels2 = branch_channels
        if(rem!=0):
            branch_channels2 += 1
            rem-=1
        branch_channels3 = branch_channels
        if(rem!=0):
            branch_channels3 += 1
            rem-=1
        branch_channels4 = branch_channels
        if(rem!=0):
            branch_channels4 += 1
            rem-=1
        # print(branch_channels1, branch_channels2, branch_channels3, branch_channels4)
        # Branch 1: 1x1 conv
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels1, kernel_size=1),
            nn.BatchNorm2d(branch_channels1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 conv followed by 3x3 conv
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels2, kernel_size=1),
            nn.Conv2d(branch_channels2, branch_channels2, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 conv followed by 5x5 conv
        # Note: Adjust the number of intermediate channels as necessary
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels3, kernel_size=1),  # Reduce channels for the 5x5 conv
            nn.Conv2d(branch_channels3, branch_channels3, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels3),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 max pooling followed by 1x1 conv
        self.branchPool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels4, kernel_size=1),
            nn.BatchNorm2d(branch_channels4),
            nn.ReLU(inplace=True)
        )
        
        # Ensure the output of the module matches the desired out_channels
        # This could involve concatenating the outputs of each branch and then
        # applying a 1x1 convolution to achieve the correct number of output channels
        self.adjust_channels = nn.Conv2d(out_channels, out_channels, kernel_size=1) if out_channels != branch_channels * 4 else None

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branchPool(x)
        
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        if self.adjust_channels is not None:
            outputs = self.adjust_channels(outputs)
        
        return outputs
    
class ResidualBlock_CustomAudio(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResidualBlock_CustomAudio, self).__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 is not None:
            identity = self.conv3(identity)
        out += identity
        return self.relu(out)
    
class InceptionModule_CustomAudio(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule_CustomAudio, self).__init__()
        
        # Adjust for 1D convolution
        rem = out_channels % 4
        branch_channels = out_channels // 4
        branch_channels1 = branch_channels + (1 if rem > 0 else 0)
        branch_channels2 = branch_channels + (1 if rem > 1 else 0)
        branch_channels3 = branch_channels + (1 if rem > 2 else 0)
        branch_channels4 = branch_channels
        
        # Branch 1: 1x1 conv (equivalent in 1D)
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels1, kernel_size=1),
            nn.BatchNorm1d(branch_channels1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 conv followed by 3x3 conv
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels2, kernel_size=1),
            nn.Conv1d(branch_channels2, branch_channels2, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels2),
            nn.ReLU(inplace=True)
        )
        
        # For simplicity, replacing 5x5 convolutions with 3x3 as 5x5 is less common in 1D
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels3, kernel_size=1),
            nn.Conv1d(branch_channels3, branch_channels3, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels3),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: Max pooling followed by 1x1 conv
        self.branchPool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels4, kernel_size=1),
            nn.BatchNorm1d(branch_channels4),
            nn.ReLU(inplace=True)
        )

        # Adjusting channels if necessary is more complex in 1D due to variable sequence lengths,
        # often not required if concatenation results in the correct number of channels

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branchPool(x)
        
        # Concatenate the outputs of each branch along the channel dimension
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        return outputs


    
def calculate_reduced_channels(channels, reduction=0.35):
    """Calculate the reduced number of channels."""
    return math.ceil(channels * (1 - reduction))

        
class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

        ### IMAGE

        initial_channels_image = 3

        self.num_classes_image = 10

        # Define the first sequence of Residual Blocks
        self.res1_image = ResidualBlock_CustomImage(initial_channels_image)
        self.res2_image = ResidualBlock_CustomImage(initial_channels_image)
        
        # Define the first Inception Block and reduce channels
        start_channels_image = 64
        inc1_out_channels_image = calculate_reduced_channels(start_channels_image)
        self.inc1_image = InceptionModule_CustomImage(initial_channels_image, inc1_out_channels_image)

        # Subsequent blocks
        inc2_out_channels_image = calculate_reduced_channels(inc1_out_channels_image)
        self.inc2_image = InceptionModule_CustomImage(inc1_out_channels_image, inc2_out_channels_image)

        # Residual Block maintains channel size after the second inception module
        self.res3_image = ResidualBlock_CustomImage(inc2_out_channels_image, inc2_out_channels_image)

        # Continuing with further inception blocks
        inc3_out_channels_image = calculate_reduced_channels(inc2_out_channels_image)
        self.inc3_image = InceptionModule_CustomImage(inc2_out_channels_image, inc3_out_channels_image)

        # Further blocks follow the pattern
        inc4_out_channels_image = calculate_reduced_channels(inc3_out_channels_image)
        self.inc4_image = InceptionModule_CustomImage(inc3_out_channels_image, inc4_out_channels_image)

        self.res4_image = ResidualBlock_CustomImage(inc4_out_channels_image, inc4_out_channels_image)

        inc5_out_channels_image = calculate_reduced_channels(inc4_out_channels_image)
        self.inc5_image = InceptionModule_CustomImage(inc4_out_channels_image, inc5_out_channels_image)

        self.res5_image = ResidualBlock_CustomImage(inc5_out_channels_image, inc5_out_channels_image)

        inc6_out_channels_image = calculate_reduced_channels(inc5_out_channels_image)
        self.inc6_image = InceptionModule_CustomImage(inc5_out_channels_image, inc6_out_channels_image)

        # Final sequence leading to the classifier
        self.avgpool_image = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_image = nn.Sequential(
            nn.Linear(inc6_out_channels_image, 256),  # Adjusted based on the final channel size
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes_image)
        )

        ### AUDIO

        initial_channels_audio = 1  # Mono audio
        self.num_classes_audio = 35  # SpeechCommands has 35 classes

        # Define the first sequence of Residual Blocks for 1D
        self.res1_audio = ResidualBlock_CustomAudio(initial_channels_audio, 128)  # Example channel size, adjust as needed
        self.res2_audio = ResidualBlock_CustomAudio(128, 256)

        # Define the first Inception Block and reduce channels for 1D
        start_channels_audio = 256
        inc1_out_channels_audio = calculate_reduced_channels(start_channels_audio)
        self.inc1_audio = InceptionModule_CustomAudio(256, inc1_out_channels_audio)

        # Subsequent blocks
        inc2_out_channels_audio = calculate_reduced_channels(inc1_out_channels_audio)
        self.inc2_audio = InceptionModule_CustomAudio(inc1_out_channels_audio, inc2_out_channels_audio)

        # Residual Block maintains channel size after the second inception module
        self.res3_audio = ResidualBlock_CustomAudio(inc2_out_channels_audio, inc2_out_channels_audio)

        # Continuing with further inception blocks
        inc3_out_channels_audio = calculate_reduced_channels(inc2_out_channels_audio)
        self.inc3_audio = InceptionModule_CustomAudio(inc2_out_channels_audio, inc3_out_channels_audio)

        # Further blocks follow the pattern
        inc4_out_channels_audio = calculate_reduced_channels(inc3_out_channels_audio)
        self.inc4_audio = InceptionModule_CustomAudio(inc3_out_channels_audio, inc4_out_channels_audio)

        self.res4_audio = ResidualBlock_CustomAudio(inc4_out_channels_audio, inc4_out_channels_audio)

        inc5_out_channels_audio = calculate_reduced_channels(inc4_out_channels_audio)
        self.inc5_audio = InceptionModule_CustomAudio(inc4_out_channels_audio, inc5_out_channels_audio)

        self.res5_audio = ResidualBlock_CustomAudio(inc5_out_channels_audio, inc5_out_channels_audio)

        inc6_out_channels_audio = calculate_reduced_channels(inc5_out_channels_audio)
        self.inc6_audio = InceptionModule_CustomAudio(inc5_out_channels_audio, inc6_out_channels_audio)

        # Final sequence leading to the classifier
        # Adjust adaptive pooling to 1D
        self.avgpool_audio = nn.AdaptiveAvgPool1d(1)  # Pools across the temporal dimension
        self.classifier_audio = nn.Sequential(
            nn.Linear(inc6_out_channels_audio, 256),  # Adjust based on the final channel size
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes_audio)
        )

    def forward(self, x):
        if(len(x.shape)==4):
            x = self.res1_image(x)
            x = self.res2_image(x)
            x = self.inc1_image(x)
            x = self.inc2_image(x)
            x = self.res3_image(x)
            x = self.inc3_image(x)
            x = self.inc4_image(x)
            x = self.res4_image(x)
            x = self.inc5_image(x)
            x = self.res5_image(x)
            x = self.inc6_image(x)
            x = self.avgpool_image(x)
            x = torch.flatten(x, 1)
            x = self.classifier_image(x)
        else:
            x = self.res1_audio(x)
            x = self.res2_audio(x)
            x = self.inc1_audio(x)
            x = self.inc2_audio(x)
            x = self.res3_audio(x)
            x = self.inc3_audio(x)
            x = self.inc4_audio(x)
            x = self.res4_audio(x)
            x = self.inc5_audio(x)
            x = self.res5_audio(x)
            x = self.inc6_audio(x)
            x = self.avgpool_audio(x)
            x = torch.flatten(x, 1)  # Flatten the features for the classifier
            x = self.classifier_audio(x)
            
        return x

def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    # device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    network = network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    
    # Write your code here
    max_accuracy = 0

    # threshold_train = 75

    # min_epoch_train = 5

    for epoch in range(EPOCH):
        # start = time.time()
        network.train()  # Set model to training mode
        ### Check if dataloader is instance of ImageDataset or AudioDataset
        if(isinstance(dataloader.dataset, ImageDataset)):
            train_correct = 0
            train_total = 0
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # if(i%100==0):
                #     print(f"{i}/{len(dataloader)}")
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = 100 * train_correct / train_total
            epoch_loss = running_loss / len(dataloader)
            print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
                epoch,
                epoch_loss,
                train_accuracy
            ))
            if(train_accuracy>max_accuracy):
                max_accuracy = train_accuracy
                torch.save(network.state_dict(), "model_checkpoint.pth")
            # end = time.time()
            # print("Time taken for epoch: ", end - start)
            # if(train_accuracy>threshold_train and epoch>min_epoch_train):
            #     break
        else:
            # start = time.time()
            # network.train()  # Set model to training mode
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0  # Keep track of total predictions to calculate accuracy
            
            for i, (waveforms, labels) in enumerate(dataloader):
                # if(i%100==0):
                #     print(f"{i}/{len(dataloader)}")
                waveforms, labels = waveforms.to(device), labels.to(device)
                # print(len(waveforms.shape))
                optimizer.zero_grad()  # Reset the gradients
                outputs = network(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Calculate and print the average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = (correct_predictions / total_predictions) * 100
            # torch.save(network.state_dict(), "model_checkpoint.pth")
            print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
                epoch,
                epoch_loss,
                epoch_accuracy
            ))
            if(epoch_accuracy>max_accuracy):
                max_accuracy = epoch_accuracy
                torch.save(network.state_dict(), "model_checkpoint.pth")
            # end = time.time()
            # print("Time taken for epoch: ", end - start)
            # if(epoch_accuracy>threshold_train and epoch>min_epoch_train):
            #     break

        

    """
    Only use this print statement to print your epoch loss, accuracy
    print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    network = network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    
    # Write your code here
    ### Load the saved model

    # threshold_train = 80

    # min_epoch_train = 2

    network.load_state_dict(torch.load("model_checkpoint.pth"))
    max_accuracy = 0
    for epoch in range(EPOCH):

        if(isinstance(dataloader.dataset, ImageDataset)):
            train_correct = 0
            train_total = 0
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # if(i%100==0):
                #     print(f"{i}/{len(dataloader)}")
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = 100 * train_correct / train_total
            epoch_loss = running_loss / len(dataloader)
            print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
                epoch,
                epoch_loss,
                train_accuracy
            ))
            if(train_accuracy>max_accuracy):
                max_accuracy = train_accuracy
                torch.save(network.state_dict(), "model_checkpoint.pth")
            # end = time.time()
            # print("Time taken for epoch: ", end - start)
            # if(train_accuracy>threshold_train and epoch>min_epoch_train):
            #     break
        else:
            # start = time.time()
            # network.train()  # Set model to training mode
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0  # Keep track of total predictions to calculate accuracy
            
            for i, (waveforms, labels) in enumerate(dataloader):
                # if(i%100==0):
                #     print(f"{i}/{len(dataloader)}")
                waveforms, labels = waveforms.to(device), labels.to(device)
                # print(len(waveforms.shape))
                optimizer.zero_grad()  # Reset the gradients
                outputs = network(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Calculate and print the average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = (correct_predictions / total_predictions) * 100
            # torch.save(network.state_dict(), "model_checkpoint.pth")
            print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
                epoch,
                epoch_loss,
                epoch_accuracy
            ))
            if(epoch_accuracy>max_accuracy):
                max_accuracy = epoch_accuracy
                torch.save(network.state_dict(), "model_checkpoint.pth")
            # end = time.time()
            # print("Time taken for epoch: ", end - start)
            # if(epoch_accuracy>threshold_train and epoch>min_epoch_train):
            #     break
        
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    
    # Write your code here
    network.load_state_dict(torch.load("model_checkpoint.pth"))

    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")

    if(isinstance(dataloader.dataset, ImageDataset)):
        test_accuracy = 0
        correct = 0
        total = 0
        loss = None
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = network(images)
                _, predicted = torch.max(outputs.data, 1)

                # calculate loss
                loss = criterion(outputs, labels)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total

        print("[Loss: {}, Accuracy: {}]".format(loss,test_accuracy))

    else:
        test_accuracy = 0
        correct = 0
        total = 0
        loss = None
        with torch.no_grad():
            for waveforms, labels in dataloader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                outputs = network(waveforms)
                _, predicted = torch.max(outputs.data, 1)

                # calculate loss
                loss = criterion(outputs, labels)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total

        print("[Loss: {}, Accuracy: {}]".format(loss,test_accuracy))

    """
    Only use this print statement to print your loss, accuracy
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """
    
    