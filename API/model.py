import torch
import torch.nn as nn
import torchvision.models as models

def load_model(model_path, num_classes, device=None):
    if device is None:
        device = torch.device('cpu')
    
    model = models.densenet121(weights=None)

    
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model