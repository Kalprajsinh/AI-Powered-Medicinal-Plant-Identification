from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0)