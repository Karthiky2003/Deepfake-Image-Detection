This repository implements a hybrid deepfake detection system combining Convolutional Neural Networks (CNN) and Vision Transformers (ViT). The model is trained to classify images as either real or deepfake based on a dataset consisting of real and fake images.

Requirements
To run the code, you'll need the following Python libraries:

torch (PyTorch)

torchvision

timm

PIL (Pillow)

opencv-python

tqdm

You can install the required libraries using pip:

bash
Copy
Edit
pip install torch torchvision timm pillow opencv-python tqdm
Project Structure
The project contains the following files:

deepfake_detector.py: The main script containing the dataset class, model architecture, and functions for training and inference.

README.md: This file, which explains how to set up and run the project.

Dataset
The model requires two directories containing real and fake images. The images should be in .jpg or .png format.

real_images_folder: Directory containing real images (e.g., images of actual people).

fake_images_folder: Directory containing fake images (e.g., deepfake-generated images).

Example Directory Structure:
markdown
Copy
Edit
project/
│
├── real_images_folder/
│   ├── real_01.jpg
│   ├── real_02.jpg
│   └── ...
│
└── fake_images_folder/
    ├── fake_01.jpg
    ├── fake_02.jpg
    └── ...
Model Architecture
The model uses two main components:

ResNet-50 (CNN): Used to extract visual features from the images.

Vision Transformer (ViT): Used to capture global image patterns that might indicate deepfake manipulation.

The features from both the CNN and ViT models are concatenated and passed through a fully connected layer for classification.

Final Classifier:
Input: 2048 (CNN) + 768 (ViT) features

Output: A binary classification (Real or Deepfake)

Training
To train the model, you need to run the train_model function. The training process includes the following steps:

Dataset Loading: Loads real and fake images from the provided directories.

Transformations: Images are resized to 224x224 and normalized.

Model Training: The model is trained using the Adam optimizer with binary cross-entropy loss.

Saving the Model: After training, the model is saved as deepfake_detector.pth.

Run Training:
python
Copy
Edit
real_images_folder = "path/to/real/images"
fake_images_folder = "path/to/fake/images"
train_model(real_images_folder, fake_images_folder, num_epochs=10, batch_size=8, learning_rate=1e-4)
Training Output:
During training, the loss will be printed for each epoch, and a progress bar will be displayed for each batch.
