import torch
import torch.nn as nn
import torchvision.models as models

def load_model(model_path, num_classes):
    model = models.densenet121(pretrained=False)
    
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model