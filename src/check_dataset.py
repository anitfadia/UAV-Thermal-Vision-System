
from pathlib import Path

# Automatically find the SIP project folder
ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset" / "images" / "thermal dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

print("Dataset path:", DATASET)
print("Dataset folder exists:", DATASET.exists())


def check_split(split_name):
    split = DATASET / split_name
    images_dir = split / "images"
    labels_dir = split / "labels"

    print(f"\n--- {split_name.upper()} DATASET ---")
    print("Images folder exists:", images_dir.exists())
    print("Labels folder exists:", labels_dir.exists())

    if not images_dir.exists() or not labels_dir.exists():
        print("Folder missing. Check the dataset structure.")
        return

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(labels_dir.glob("*.txt"))

    image_names = {p.stem for p in images}
    label_names = {p.stem for p in labels}

    print("Number of images:", len(images))
    print("Number of label files:", len(labels))
    print("Images without matching labels:", len(image_names - label_names))
    print("Labels without matching images:", len(label_names - image_names))

    # Show a sample annotation so we can identify its format
    if labels:
        sample = labels[0]
        print("\nSample label file:", sample.name)
        print("Sample annotation:")
        print(sample.read_text(encoding="utf-8")[:500])


check_split("train")
check_split("test")