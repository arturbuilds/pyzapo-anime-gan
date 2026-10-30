# PyZapo Anime Face GAN

A fully connected Generative Adversarial Network (GAN) for generating 28×28 anime face images, built with the PyZapo machine learning framework.

## Features

- Fully connected GAN architecture
- Custom PyZapo neural network layers
- Pre-trained generator weights
- Random anime face generation
- Lightweight inference

## Architecture

### Generator
- Linear(100, 512)
- LeakyReLU
- Linear(512, 2352)
- Tanh

### Discriminator
- Linear(2352, 128)
- LeakyReLU
- Linear(128, 1)
- Sigmoid

## Model Information

| Parameter | Value |
|---|---|
| Framework | PyZapo |
| Model | Fully Connected GAN |
| Image resolution | 28×28 |
| Image channels | RGB |
| Latent dimension | 100 |
| Training epochs | 500 |
| Batch size | 64 |
| Learning rate | 0.0002 |

## Installation

```bash
pip install -r requirements.txt
```

## Pre-trained Weights

Place the downloaded generator weights in:

```text
weights/anime_generator_v1.npz
```

The dataset is not required for image generation.

## Generate Images

```bash
python generator.py
```

Generated images will be saved in the `generated/` directory.

## Training

To train the GAN from scratch, provide your own `anime_images.npy` dataset in the project root and run:

```bash
python train.py
```

Training images and final generator weights will be saved locally.

## License

See LICENSE for details.
