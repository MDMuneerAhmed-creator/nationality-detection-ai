"""
Custom CNN Architecture Definitions Built Completely From Scratch.

STRICT CONSTRAINTS COMPLIANCE:
- NO Pretrained Models (ResNet, MobileNet, EfficientNet, VGG, YOLO, FaceNet, CLIP, etc.)
- NO Pretrained Weights or Transfer Learning
- ALL layers initialized with random weights (He / Kaiming Normal)
- Architecture:
  Block 1: Conv2D(32) -> BatchNorm2d -> ReLU -> MaxPool2d
  Block 2: Conv2D(64) -> BatchNorm2d -> ReLU -> MaxPool2d
  Block 3: Conv2D(128) -> BatchNorm2d -> ReLU -> MaxPool2d
  Block 4: Conv2D(256) -> BatchNorm2d -> ReLU -> MaxPool2d
  Block 5: Conv2D(512) -> BatchNorm2d -> ReLU -> MaxPool2d
  Flatten
  Dense(1024) -> ReLU -> Dropout(0.5)
  Dense(512) -> ReLU -> Dropout(0.3)
  Dense(num_classes) -> Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NATIONALITIES = [
    "Indian", "American", "African", "Japanese", "German"
]

EMOTIONS = [
    "Happy", "Neutral", "Surprise", "Sad", "Angry"
]

AGE_GROUPS = [
    "0-18", "19-35", "36-50", "51+"
]

DRESS_COLORS = [
    "Red", "Blue", "Yellow", "Black", "Green", "White", "Other"
]

# Input tensor resolution: 224x224x3
# After 5 MaxPool2d(2, 2) operations:
# 224 -> 112 -> 56 -> 28 -> 14 -> 7
# Feature map shape before flatten: 512 * 7 * 7 = 25088

class CustomDeepCNN(nn.Module):
    """
    Deeper 5-Block Convolutional Neural Network designed for high accuracy from scratch.
    All parameters initialized with He/Kaiming Normal initialization.
    """
    def __init__(self, num_classes: int, input_channels: int = 3, model_name: str = "CustomDeepCNN"):
        super(CustomDeepCNN, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Block 1: Conv(32) -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: Conv(64) -> BatchNorm -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: Conv(128) -> BatchNorm -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4: Conv(256) -> BatchNorm -> ReLU -> MaxPool
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Block 5: Conv(512) -> BatchNorm -> ReLU -> MaxPool
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # Dense Classifier Layers
        # 512 channels * 7 * 7 spatial grid = 25088
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(0.3)

        self.out = nn.Linear(512, num_classes)

        # Initialize with He (Kaiming) Normal
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Block 5
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Fully Connected Classifier
        x = self.flatten(x)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        logits = self.out(x)
        return F.softmax(logits, dim=-1)

def create_nationality_model() -> CustomDeepCNN:
    return CustomDeepCNN(num_classes=len(NATIONALITIES), model_name="Nationality_CNN")

def create_emotion_model() -> CustomDeepCNN:
    return CustomDeepCNN(num_classes=len(EMOTIONS), model_name="Emotion_CNN")

def create_age_model() -> CustomDeepCNN:
    return CustomDeepCNN(num_classes=len(AGE_GROUPS), model_name="Age_CNN")

def create_dress_color_model() -> CustomDeepCNN:
    return CustomDeepCNN(num_classes=len(DRESS_COLORS), model_name="Dress_Colour_CNN")
