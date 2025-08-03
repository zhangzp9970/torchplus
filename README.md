![torchplus](./doc/images/torchplus.svg)

# torchplus

torchplus is a utilities library that extends pytorch and torchvision

## Install

Install [Anaconda](https://www.anaconda.com/) or [Python](https://www.python.org/)

Install Pytorch using the commands from the [official pytorch page](https://pytorch.org/). As pytorch removes their official conda channel since 2.6, we will also not update the conda channel.

Install torchplus using pypi `pip install tplus` (recommended) 

deprecated conda channel `conda install torchplus -c zhangzp9970`


## Features

* torchplus.data
  * DataZ -- The self-defined data format **.dataz** that combines the images (in tensor format) with its properties. Useful in tasks that requires different properties or attributes of the images.
* torchplus.datasets
  * KSDD -- [KolektorSDD](http://go.vicos.si/kolektorsdd) dataset loader, compatible with other torchvision APIs.
  * KSDD2 --[KolektorSDD2](http://go.vicos.si/kolektorsdd2) dataset loader, compatible with other torchvision APIs.
  * FlatFolder -- A dataset class to handle flat folders with no sub directories, such as CelebA and FFHQ.
  * PreProcessFolder -- Preload the image data from disk into system memory to accelerate training.
  * DataZFolder -- Data loader to load the .dataz data arranged in folder tree way.
* torchplus.distributed
  * FederatedAverage -- Implements the federated average algorithm.
* torchplus.models
  * ResNetFE and resnet*fe feature extractor of the Deep Residual Network (ResNet), with optional pretrained weights can be loaded.
* torchplus.math
  * pi -- The pi=3.1415.
* torchplus.nn
  * MSEWithWeightLoss -- Weighted mean square error function.
  * PixelLoss -- Calculates the differences of pixels in two batches of images.
  * PowerAmplification -- The amplification function proposed in paper "[Analysis and Utilization of Hidden Information in Model Inversion Attacks](https://doi.org/10.1109/TIFS.2023.3295942)"
* torchplus.nn.functional
  * Functional implenmentation of MSEWithWeightLoss and PixelLoss, just like torch.nn.functional.
* torchplus.transforms
  * Crop -- Packages the torchvision.transforms.functional.crop function into a class, make it compatible with other transforms in torchvision.transforms.transforms.
* torchplus.utils
  * Init -- Initialize all useful functions, make it easy to write code. Specifically, initialize all seeds, make log directory, initialize tensorboard, automatically backup the current file into the log directory, initialize pytorch profiler. It can returns the log directory, the tensorboard writer object, the pytorch profilier object, the output device(CPU or CUDA).
  * BaseAccuracy -- Base class to calculate accuracy for the whole dataset or accuracies for each class.
  * ClassificationAccuracy -- Calculate classification accuracy for the classification task.
  * class_split -- Split the dataset into subdatasets based on the classes, like torch.utils.data.random_split(), which randomly split the datasets.
  * save_excel -- Save a tensor to excel spreadsheet, make it easy to analysis the data.
  * save_csv -- Save a tensor to csv sheet.
  * save_image2 -- Enhance the save_image function in torchvision by intelligent creating directories.
  * read_image_to_tensor -- Read a image into a tensor. Since the torchvision.io.read_image() can only read jpg and png images, which are not so useful in practice.
  * hash_code -- Return a hash code for a object, useful when you want to produce time-variant filenames without changing the code.
  * MMD -- Calculate the Maximum Mean Discrepancy of two batches of tensors. Code from the internet.
  * model_size -- Calculate the size of the model. Refer to: [https://discuss.pytorch.org/t/finding-model-size/130275/2](https://discuss.pytorch.org/t/finding-model-size/130275/2).
* continue developing...

## Acknowledgements

Inspired by [easydl](https://pypi.org/project/easydl/) project from Tsinghua University

## License

Copyright © 2021-2025 Zeping Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
