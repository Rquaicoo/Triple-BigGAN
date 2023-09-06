# Triple-BigGAN Implementation README

This repository contains an implementation of Triple-BigGAN with several variations and experimentation notebooks.

## Contents

- [Project Structure](#project-structure)
- [Libraries Used](#libraries-used)
- [Implementation Details](#implementation-details)
- [Running the Code](#running-the-code)
- [Original Paper](#original-paper)

## Project Structure

The project is organized as follows:

- **classifier.py**: Contains the code for the classifier.
- **generator.py**: Contains the generator code without spectral normalization.
- **generator_with_spectral_normalization.py**: Contains the generator code with spectral normalization.
- **discriminator.py**: Concatenates the embedding class with a layer above and passes the result through a dense layer.
- **discriminator_with_inner_product.py**: Uses an inner product rather than the approach described above.
- **notebook1.ipynb**, **notebook2.ipynb**, and **notebook3.ipynb**: Contain different variations of the generator, discriminator, and classifier experimentation. These notebooks document the experiments with the Triple-BigGAN model.
- **test_notebook.py**: Contains tests using downloaded weights.
- **utils.py** and **SpectralNormalization.py**: Contain utility functions used in the experimentation.
- **main.py**: Includes a `__main__` function that can be used to run the code. You can adjust parameters such as batch sizes, the number of epochs, and sample sizes for training in this file.

## Libraries Used

The following libraries were used in this project:

- TensorFlow: A deep learning framework for building and training neural networks.
- tensorflow-gan: An extension library for TensorFlow designed for training GANs and related models.
- Matplotlib: A Python plotting library used for data visualization.
- NumPy: A fundamental package for scientific computing in Python, used for array and matrix operations.
- Wandb (Weights and Biases): A tool for visualizing and tracking machine learning experiments, helpful for monitoring the training process and results.

## Implementation Details

### Pixel Normalization

In our implementation, we applied Pixel Normalization as a preprocessing step for GAN images. Pixel Normalization helps stabilize the training process by normalizing pixel values to have zero mean and unit variance.

### One-Hot Encoding

We used One-Hot Encoding for labels in our implementation, unlike the original paper, which may have used categorical labels directly. One-Hot Encoding converts class labels into binary matrices, a common practice for neural network classification tasks.

### Training on Sampled Image Subsets

Rather than using the entire CIFAR-10 dataset (60,000 images), we trained our model on two subsets: 1,000 images and 10,000 images. This choice reduced training time while still achieving reasonable results.

### Loss Functions

We employed different loss functions compared to the original paper:
- Categorical Cross-Entropy: Used for the classifier (discriminator) to distinguish between different classes.
- Binary Cross-Entropy: Used for the generator and discriminator networks in GANs to distinguish real and generated samples.

These adaptations were made to suit our dataset and problem requirements.

## Running the Code

You can run the code by adjusting parameters in the `main.py` file. Experiment with batch sizes, epochs, and sample sizes according to your specific goals.

## Original Paper

The original paper, titled "Triple-BigGAN: Semi-supervised generative adversarial networks for image synthesis and classification on sexual facial expression recognition," is available in the PDF document included in this repository.

For more details, refer to the original paper for a comprehensive understanding of the Triple-BigGAN model.

