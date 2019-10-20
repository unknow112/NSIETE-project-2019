# Neural networks @ FIIT - Project Proposal 🖼
## Organisational:
**Authors:** Martin Tonhauzer, Viktor Valaštín

**Course Supervisor:** Ing. Michal Farkaš

**Academic Year:** 2019/2020, Winter Semester
# Image upscaling

## Motivation

You remember those images with really bad resolution since the pixels started to matter? Well now you dont need to fear them anymore!
Jokes aside.
The goal of this project is to create [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network) to be able to upscale image resolution. Ideally it should be able to process GIFs and short videos.
We see huge potential of this technology in streaming high quality video content while preserving network bandwidth.
You could encode video stream at lower resolution and then upscale it at end device, without noticeable quality loss.


## Related Work
There ares several papers related to image upscaling

 - [Photo-Realistic Single Image Super-Resolution Using GAN](https://arxiv.org/abs/1609.04802)
 - [High-Quality Face Image Super-Resolution Using Conditional GAN](https://arxiv.org/pdf/1707.00737.pdf)

And you can find real world application in:
 - [Ray Tracing upscaling in video games](https://www.nvidia.com/en-us/geforce/news/dlss-control-and-beyond/)
 - [Movie upscaling](https://www.provideocoalition.com/videogorillas-bigfoot-super-resolution-converts-films-from-native-480p-to-4k/)

Also nVidia offers implementation of such use case as part of [NGX](https://developer.nvidia.com/rtx/ngx) package. We might be able to use it as reference when comparing the quality of our model.

## Datasets

We are going to create our own image dataset. We download images from photobanks such as [Pixabay](https://pixabay.com), so we will have a lot of different types of images. After that we downscale them and then use them as input for our neural network.

## High-Level Solution Proposal
We personally think that our solution will consist couple of steps:

1. Creating dataset:
 - Download a huge dataset of images from photobank
 - Downscale them to appropriate dimensions (we could also downscale image on multiple levels (like from 1024x1024 to 256x256 and 128x128 images)). Use various lossy compression algorithms
 - create train, test and verify datasets.
2. Training and engineering the model.

Additionally we would try to use similar approach for GIFs and videos.
