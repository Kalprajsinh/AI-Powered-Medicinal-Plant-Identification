class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                     if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            self.image_paths.extend(images)
            self.labels.extend([self.class_to_idx[class_name]] * len(images))

        print(f"Dataset: {len(self.image_paths)} images, {len(self.classes)} classes from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = PlantDataset(root_dir=train_dir, transform=train_transform)
test_dataset = PlantDataset(root_dir=test_dir, transform=test_transform)