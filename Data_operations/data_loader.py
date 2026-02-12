dataset_path = "/content/IndoHerb"

folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print(f"Total plant species folders: {len(folders)}")
print(folders)
print(f"Total images: {sum([len(os.listdir(os.path.join(dataset_path, f))) for f in folders])}")

dataset_path = "/content/IndoHerb"
base_dir = "/content/IndoHerb_split"

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)