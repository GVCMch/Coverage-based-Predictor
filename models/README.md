# models/

Stores target DNN weights. Gitignored.

```
models/
├── CIFAR-10/
│   ├── vgg13_bn_cifar10_best.pt
│   ├── vgg16_bn_cifar10_best.pt
│   ├── resnet34_cifar10_best.pt
│   ├── resnet50_cifar10_best.pt
│   ├── densenet121_cifar10_best.pt
│   └── mobilenet_v2_cifar10_best.pt
├── MNIST/
├── Fashion-MNIST/
└── TinyImageNet/
```

Naming: `{model}_{dataset_tag}_best.pt`  
Tags: `cifar10`, `mnist`, `fashionmnist`, `tinyimagenet`

Loaded by `shared_config.py:build_model()`.
