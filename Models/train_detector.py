import os
import torch
from pathlib import Path
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = BASE_DIR / "dataset" / "dataset.yaml"

def main():
    if not DATASET_YAML.exists():
        print(f"Error: dataset.yaml not found at {DATASET_YAML}.")
        print("Please place your raw images in 'Dataset/raw_images' and labels in 'Dataset/raw_labels',")
        print("then run 'python Dataset/prepare_dataset.py' to split and prepare the dataset.")
        return

    # Initialize YOLOv8 Small model (pretrained on COCO)
    model = YOLO("yolov8s.pt")

    # Detect device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device for training: {device}")

    # Train model
    print(f"Starting YOLOv8 training on dataset using {DATASET_YAML}...")
    model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        patience=10,
        cos_lr=True,
        workers=2,
        exist_ok=True
    )
    print("\nTraining completed successfully!")
    print("Your AI model weights are saved at: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
