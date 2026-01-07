import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
from dotenv import load_dotenv

load_dotenv()

IMG_SIZE = (224, 224)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),              
])

def convert_dataset(input_dir, output_dir, prefix):
    class_names = sorted(os.listdir(input_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    images = []
    labels = []

    for cls in class_names:
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in tqdm(os.listdir(cls_path), desc=f"Processing {cls}"):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)   

                images.append(img_tensor)
                labels.append(class_to_idx[cls])

            except Exception as e:
                print(f"Skipped {img_path}: {e}")

    X = torch.stack(images)   
    y = torch.tensor(labels)

    torch.save(X, os.path.join(output_dir, f"{prefix}_X.pt"))
    torch.save(y, os.path.join(output_dir, f"{prefix}_y.pt"))

    with open(os.path.join(output_dir, f"{prefix}_classes.json"), "w") as f:
        json.dump(class_names, f)

    print(f"Saved {len(X)} samples")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    tensor_root = os.getenv("TENSOR_DIR")
    
    convert_dataset(os.getenv("TRAIN_DIR"), output_dir=tensor_root, prefix="train")
    convert_dataset(os.getenv("TEST_DIR"), output_dir=tensor_root, prefix="test") 
