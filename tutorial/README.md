# OpenFWI Documentation
> An Open-source Large-scale Multi-structural Benchmark Datasets for Full Waveform Inversion

[![Made with latest Docsify](https://img.shields.io/npm/v/docsify/latest?label=docsify)](https://docsify.js.org/)

## About

OpenFWI(v1.0) datasets include eleven _2D_ datasets and one _3D_ dataset. 

OpenFWI provides benchmarks on these datasets with four deep learning methods: _InversionNet_, _VelocityGAN_, _UPFWI_ for 2D and _InversionNet_ for 3D. Currently, _InversionNet_ and _VelocityGAN_ are fully suported on 11 2D datasets. The rest will be released upon approval by Los Alamos National Lab.

This tutorial is designed for beginners on data-driven FWI, knowledge on [pytorch](https://pytorch.org) is preferred but not necessary.

## Features

- Multi-scale
- Multi-structural
- Multi-subsurface-complexity

## Prepare Data
First download any dataset from the [website](https://openfwi-lanl.github.io/docs/data.html#vel) and unzip it into your local directory.

### Load a pair of velocity map and seismic data
For any dataset in _Vel, Fault, Style_  family, the data is saved as `.npy` files, each file contains a batch of 500 samples. `datai.npy` refers to the `i-th` sample of seismic data. To load data and check:
```bash
import numpy as np
# load seismic data
seismic_data = np.load('data1.npy')
print(seismic_data.shape) #(500,5,1000,70)
# load velocity map
velocity_map = np.load('model1.npy')
print(velocity_map.shape) #(500,1,70,70)
```

For Kimberlina-$CO_2$, the difference is each file only contains one sample, which has shape of (401,141) for velocity maps and (9,1251,101) for seismic data.

### Visualization

### Prepare training and testing set
Note that there are many ways of organizing training and testing dataset, as long as it is compatible with the [DataLoader module](https://pytorch.org/docs/stable/data.html) in pytorch. Whichever way you choose, please refer to the following table for the train/test split.

| Dataset      | Train / test Split | Corresponding `.npy` files |
| ----------- | ----------- | ------------ |
| Vel Family     | 24k / 6k     | data(model)1-48.npy / data(model)49-60.npy |
| Fault Family   | 48k / 6k     | data(model)1-96.npy / data(model)97-108.npy |
| Style Family   | 60k / 7k     | data(model)1-120.npy / data(model)121-134.npy |
<!-- | Kimberlina-$$CO_2$$  | 15k / 4430 | 
| Kimberlina-3D V1     | 1664 / 163 |  -->

A convinient way of loading the data is to use a `.txt` file containing the _location+filename_ of all `.npy` files, parse each line of the `.txt` file and push to the dataloader. Take **flatvel-A** as an exmaple, we create `flatvel-a-train.txt`, organized as the follows, and same for `flatvel-a-test.txt`. 
```bash
Dataset_directory/data1.npy
Dataset_directory/data2.npy
...
Dataset_directory/data48.npy
```

**To save time, you can download all the text files from the `split_files` folder at our [Github repo](https://github.com/lanl/openfwi) and change to your own directory.**

## Fast Training and Testing
Now we are ready to train a neural network on OpenFWI datasets. For 2D datasets with InversionNet and VelocityGAN, training with a single GPU is sufficient. For Kimberlina-3D dataset, multi-GPU is necessary considering the computation cost.

To start with, download the codes from the [Github repo](https://github.com/lanl/openfwi).

Our implementation follows the common practice of data pre-processing and loading, model training and test. If you are an expert on deep learning, follow the commands below to reproduce OpenFWI benchmarks or train and test your own model. If you are new to deep learning, no worries! Refer to for step-by-step tutorials.

### Environment setup
The following packages are required:
- pytorch v1.7.1
- torchvision v0.8.2
- scikit learn
- numpy
- matplotlib (for visualization)

If you other versions of pytorch and torchvision, please make sure they align.


### InversionNet
To train from scratch on Flatvel-A dataset with $$\ell_1$$ loss,  run the following codes:
```
python train.py -ds flatvel-a -n YOUR_DIRECTORY -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```

`-ds` specifies the dataset, `-n` creates the folder containing the saved model other log files, `-g2v` sets the coefficient of $\ell_2$ loss to be zero, `-t` and `-v` assign the training data and test data loading files.

To continue training from a saved checkpoint, run the following codes:
```
python train.py -ds flatvel-a -n YOUR_DIRECTORY -r CHECKPOINT.PTH -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```

Please refer to the details of the codes if you would like to change other parameters (*learning rate,* etc.). These commands suffice to reproduce the OpenFWI benchmarks.

The last step would be testing, where we include the visualization. Also we borrow the implementation of SSIM metric from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim). Please make sure that `pytorch-ssim.py` and `rainbow256.npy` are placed together with others.

```
python test.py -ds flatvel-a -n YOUR_DIRECTORY -m InversionNet -v flatvel_a_val.txt -r CHECKPOINT.PTH --vis -vb 2 -vsa 3
```

`--vis` enables the visualization and creates a folder with the figures, you may also change the amount of velocity maps by playing with `-vb` and `-vsa`.

### VelocityGAN
The code logic of VelocityGAN is almost identical the that of InversionNet.

To train from scratch on Flatvel-A dataset with $\ell_1$ loss, run the following codes:
```
python gan_train.py -ds flatvel-a -n YOUR_DIRECTORY -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```
To continue training from a saved checkpoint, run the following codes:
```
python gan_train.py -ds flatvel-a -n YOUR_DIRECTORY -r CHECKPOINT.PTH -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```
The command for testing is the same with InversionNet



## Step-by-step Training and Testing

We start with introducing files related to training and testing.

- `dataset_config.json` contains the generation parameters of each dataset, including the min/max velocity, size, frequency, etc. These parameters are required in the normalization and forward modelling module of UPFWI.
- `dataset.py` defines __FWIDataset__, a subclass of *torch.utils.data* that load every data sample from the text file to the dataloader.
- `network.py` contains the model architecture of InversionNet and VelocityGAN.
- `train.py` and `gan_train.py` execute the training of InversionNet and VelocityGAN respectively.
- `transforms.py` and `utils.py` provide peripheral functions needed for normalization, metric logging, loss function ,etc.
- `vis.py` supports the visualization of velocity maps and seismic data.


Next, with InversionNet as an example, we explain the main body of the training, as presented in `train.py`.

### Pytorch DataLoader
The first step is to load data and transform both velocity maps and seismic data to [-1,1], the transformation is given by:
```
transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=args.k))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])
```
where *data* is for seismic data and *label* stands for velocity maps, same for later.

For the data loading, we use the standard data loading module stemmed from *torch.utils.data.DataLoader*. We create two loaders from training data and testing data, after creating a FWIDataset instance for both.

```
 dataset_train = FWIDataset(args.train_anno, preload=True,...)
 dataloader_train = DataLoader( dataset_train, batch_size=args.batch_size, ...)
 dataset_valid = FWIDataset(args.val_anno, preload=True,...)
 dataloader_valid = DataLoader( dataset_valid, batch_size=args.batch_size, ...)
```
### Model Create
```
model = network.model_dict[args.model](...).to(device)
```

### Training
We first specify the loss function, optimization learning rate scheduler, etc. then the training is good to go. Notice that to reproduce the benchmark results, you do not have to change these as they have been set as default. However, you can play with *parse_args()* to change any parameter. 

The training process follows:
```
for each epoch:
    model = train_one_epoch(model, ...)
    loss = evaluate(model, ...)
    if loss reduces:
        save_checkpoint(model)
```

Refer to the above section for the codes to start the training.

### Testing
There are two additional modules during the testing phase: 1. Compute the loss and SSIM. 2. Visualization.

We use the implementation of SSIM metric from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim), and it's very simple to use:
```
ssim_loss = pytorch_ssim.SSIM(window_size=11)
print(f'SSIM: {ssim_loss(img_1, img_2}')
```

The visualization requires denormalization of the velocity maps, recall that they are transformed to [-1,1]. Therefore we use __tonumpy_denormalize()__ function from `transforms.py`:
```
label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'])
```
Refer to the above section for the codes to start the testing.