# datasets/

Stores image datasets. Gitignored.

```
datasets/
├── CIFAR10/
│   ├── train/{class}/*.jpg
│   └── test/{class}/*.jpg
├── MNIST/
│   ├── train/{0-9}/*.jpg
│   └── test/
├── Fashion/
└── TinyImageNet/
    └── tiny-imagenet-200/
        ├── train/
        └── val/
```

MNIST and Fashion-MNIST images should be converted to 3-channel RGB.

TinyImageNet must be downloaded manually:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d datasets/TinyImageNet/
```
