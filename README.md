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

Traing neural network for image processing is not easy task which require a lot of images, especially if we use GAN.
We stored all images to our personal [server](http://static.dthi.eu/datasets/) images with links to paths and sources are written below:

- [[div2k/]](http://static.dthi.eu/datasets/div2k/)  **[Source](https://data.vision.ee.ethz.ch/cvl/DIV2K/)**
- [[Flickr2K/]](http://static.dthi.eu/datasets/Flickr2K/) **[Source](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)**  
- [[dtd/]](http://static.dthi.eu/datasets/dtd/)  **[Source](http://www.robots.ox.ac.uk/~vgg/data/dtd/)**  
- [[google\_oid/]](http://static.dthi.eu/datasets/google_oid/) (note: only validation sample fully downloaded since whole dataset is 18TB big)  **[Source](https://storage.googleapis.com/openimages/web/index.html)**  
- [[places205/]](http://static.dthi.eu/datasets/places205/)   **[Source](http://places.csail.mit.edu/index.html)**  



## High-Level Solution Proposal
We personally think that our solution will consist couple of steps:

1. Creating dataset:
 - Download a huge images from datasources described above.
 - Crop images to square
 - Downscale them to appropriate dimensions (32x32px)
 - create train, test and verify datasets.
2. Training and engineering the model.
Additionally we would try to use similar approach for GIFs and videos.

## Project Structure

There are many python and other important files in this repository, we provide list with short description to each file:

 - [create_dataset](create_dataset.py) take list of images crop them to square and resize them to 32x32px images
 - [discriminator](discriminator.py) source code for Discriminator
 - [fetcher](fetcher.py) image dowloader from [Pixabay](https://pixabay.com)
 - [flow](flow.py) split image dataset to batches and run epochs on GAN
 - [gan](gan.py) wrapper for Generator and Discriminator
 - [generator](generator.py) source code for Generator
 - [resizer](resizer.py) Utilities for image processing
 - [analyze](analyze.ipynb) Jupyter notebook with data analysis
 - [dataset_preparation](dataset_preparation.ipynb) describes preparation of data
 - [requirements](requirements.txt) list of required libraries  


## Evaluation

We basically take generated image and visually check with expected output.
Then we discuss about what parts/shapes of image were done correctly.
If we get into point, where we are unable say distinct difference between pictures we use [structural similarity](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity) to calculate it.
