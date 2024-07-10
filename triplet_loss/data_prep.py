import os
import shutil
import random

# Set the directory where your classes are
base_dir = "datasets/face_enhanced"

# Directories for the training, validation, and test splits
train_dir = os.path.join("triplet_loss", "dataset", "train")
val_dir = os.path.join("triplet_loss", "dataset", "validation")
test_dir = os.path.join("triplet_loss", "dataset", "test")


# Function to create directories
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Creating train, validation, and test directories
create_dir(train_dir)
create_dir(val_dir)
create_dir(test_dir)

# Split ratio (e.g., 50% training, 25% validation, 25% test)
train_ratio = 0.5
val_ratio = 0.25


# Function to split images
def split_images(class_dir):
    # Create subdirectories in train, val, and test directories
    train_subdir = os.path.join(train_dir, class_dir)
    val_subdir = os.path.join(val_dir, class_dir)
    test_subdir = os.path.join(test_dir, class_dir)

    create_dir(train_subdir)
    create_dir(val_subdir)
    create_dir(test_subdir)

    # List all files in the class directory
    files = os.listdir(os.path.join(base_dir, class_dir))
    random.shuffle(files)  # Shuffle to randomize

    # Split files
    train_files = files[: int(len(files) * train_ratio)]
    remaining = files[int(len(files) * train_ratio) :]
    val_files = remaining[: int(len(remaining) * (val_ratio / (1 - train_ratio)))]
    test_files = remaining[int(len(remaining) * (val_ratio / (1 - train_ratio))) :]

    # Copy files to respective directories
    for f in train_files:
        shutil.copy(os.path.join(base_dir, class_dir, f), train_subdir)
    for f in val_files:
        shutil.copy(os.path.join(base_dir, class_dir, f), val_subdir)
    for f in test_files:
        shutil.copy(os.path.join(base_dir, class_dir, f), test_subdir)


# Process each class
for class_dir in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, class_dir)):
        split_images(class_dir)

print("Images have been split into train, validation, and test sets.")
