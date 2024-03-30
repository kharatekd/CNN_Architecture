# Player Classification: Dhoni vs. Kohli using CNN with Transfer Learning

This project focuses on binary classification of two prominent cricket players, Dhoni and Kohli, using Convolutional Neural Networks (CNN) with transfer learning. Transfer learning is employed to leverage the pre-trained ResNet architecture for feature extraction, which is then fine-tuned on the target dataset.

## Overview
The objective of this project is to build a robust model capable of distinguishing between images of MS Dhoni and Virat Kohli, two iconic cricket players. We utilize transfer learning with the ResNet architecture to expedite model training and improve classification performance.

## Dataset
The dataset consists of images of MS Dhoni and Virat Kohli collected from various sources. It is essential to have a balanced dataset with an equal number of images for both classes to avoid bias during training. Due to privacy or licensing concerns, the dataset used in this project is not provided, but any similar dataset containing images of Dhoni and Kohli can be used for experimentation.

## Implementation
Dependencies
Python 3.x
TensorFlow
Keras
Steps
## Data Preparation: Organize the dataset into two directories, one for Dhoni images and another for Kohli images. Split the dataset into training, validation, and test sets.

Model Architecture: Utilize transfer learning by loading the pre-trained ResNet model without the top (fully connected) layers. Add custom fully connected layers on top of the ResNet layers for binary classification.

## Training: 
Train the model on the training data while validating its performance on the validation set. Fine-tune the model's weights to improve classification accuracy.

## Evaluation: 
Evaluate the trained model on the test set to assess its performance metrics, such as accuracy, precision, recall, and F1-score.

## Deployment: 
Deploy the trained model for inference on new images to classify whether the input image contains Dhoni or Kohli.

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvement, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

