# torchplus

torchplus is a utilities library that extends pytorch and torchvision

[![Anaconda-Server Badge](https://anaconda.org/zhangzp9970/torchplus/badges/version.svg)](https://anaconda.org/zhangzp9970/torchplus)
[![Anaconda-Server Badge](https://anaconda.org/zhangzp9970/torchplus/badges/latest_release_date.svg)](https://anaconda.org/zhangzp9970/torchplus)
[![Anaconda-Server Badge](https://anaconda.org/zhangzp9970/torchplus/badges/latest_release_relative_date.svg)](https://anaconda.org/zhangzp9970/torchplus)
[![Anaconda-Server Badge](https://anaconda.org/zhangzp9970/torchplus/badges/platforms.svg)](https://anaconda.org/zhangzp9970/torchplus)
[![Anaconda-Server Badge](https://anaconda.org/zhangzp9970/torchplus/badges/license.svg)](https://anaconda.org/zhangzp9970/torchplus)
[![Anaconda-Server Badge](https://anaconda.org/zhangzp9970/torchplus/badges/downloads.svg)](https://anaconda.org/zhangzp9970/torchplus)
[![Anaconda-Server Badge](https://anaconda.org/zhangzp9970/torchplus/badges/installer/conda.svg)](https://conda.anaconda.org/zhangzp9970)

## Features

* torchplus.datasets
  * KSDD -- [KolektorSDD](http://go.vicos.si/kolektorsdd) dataset loader, compatible with other torchvision APIs.
  * KSDD2 --[KolektorSDD2](http://go.vicos.si/kolektorsdd2) dataset loader, compatible with other torchvision APIs.
* torchplus.nn
  * MSEWithWeightLoss -- Weighted mean square error function.
  * PixelLoss -- Calculates the differences of pixels in two batches of images.
* torchplus.nn.functional
  * Functional implenmentation of MSEWithWeightLoss and PixelLoss, just like torch.nn.functional.
* torchplus.transforms
  * Crop -- Packages the torchvision.transforms.functional.crop function into a class, make it compatible with other transforms in torchvision.transforms.transforms.
* torchplus.utils
  * Init -- Initialize all useful functions, make it easy to write code. Specifically, initialize all seeds, make log directory, initialize tensorboard, automatically backup the current file into the log directory, initialize pytorch profiler, initialize data workers using multiprocessing.cpu_count(). It can returns the log directory, the tensorboard writer object, the pytorch profilier object, the output device(CPU or CUDA), the data workers (for DataLoader)
  * BaseAccuracy -- Base class to calculate accuracy for the whole dataset or accuracies for each class.
  * ClassificationAccuracy -- Calculate classification accuracy for the classification task.
  * class_split -- split the dataset into subdatasets based on the classes, like torch.utils.data.random_split(), which randomly split the datasets.
  * save_excel -- save a tensor to excel spreadsheet, make it easy to analysis the data.
  * read_image_to_tensor -- read a image into a tensor. Since the torchvision.io.read_image() can only read jpg and png images, which are not so useful in practice.
* continue developing...

## Acknowledgements

Inspired by [easydl](https://github.com/thuml/easydl) project from Tsinghua University

## License

Copyright © 2021 Zeping Zhang

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
