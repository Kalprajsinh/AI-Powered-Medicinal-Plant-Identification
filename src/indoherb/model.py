import torch
import torch.nn as nn
import torchvision.models as models

class PlantModel:
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model or create scratch model
        if model_name == "resnet34":
            self.model = models.resnet34(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)

        elif model_name == "vgg11":
            self.model = models.vgg11_bn(pretrained=True)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        elif model_name == "scratch_cnn":
            self.model = ScratchCNN(num_classes=num_classes)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Move to device
        self.model.to(self.device)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{model_name} Trainable parameters: {trainable_params:,}")

class ScratchCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(ScratchCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 512),  # 128x128 input -> 16x16 after 3 maxpool layers
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Alternative simpler Scratch CNN (as mentioned in paper)
class SimpleScratchCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleScratchCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Layer 1: 32 filters, 3x3 kernel
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Layer 2: 64 filters, 3x3 kernel
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Layer 3: 128 filters, 3x3 kernel
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),  # Adjusted for 128x128 input
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Usage example
if __name__ == "__main__":
    # Test all models
    models_to_test = ["resnet34", "densenet121", "vgg11", "scratch_cnn"]

    for model_name in models_to_test:
        print(f"\n=== Testing {model_name} ===")
        try:
            model_wrapper = PlantModel(model_name, num_classes=100)
            print(f"Successfully created {model_name}")

            # Test forward pass
            test_input = torch.randn(2, 3, 128, 128).to(model_wrapper.device)
            output = model_wrapper.model(test_input)
            print(f"Output shape: {output.shape}")

        except Exception as e:
            print(f"Error with {model_name}: {e}")