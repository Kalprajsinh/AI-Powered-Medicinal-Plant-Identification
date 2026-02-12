# 60% train, 40% test
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path)
             if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(images)
    split_idx = int(0.6 * len(images))
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img),
                   os.path.join(train_dir, class_folder, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img),
                   os.path.join(test_dir, class_folder, img))
print("Dataset split")