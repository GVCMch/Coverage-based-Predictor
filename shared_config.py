import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms


DATASET_CONFIGS = {
    'CIFAR10': {
        'num_classes': 10,
        'image_size': 32,
        'channels': 3,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'data_dir': './datasets/CIFAR10/',
        'weight_dir': './models/CIFAR-10',
        'weight_tag': 'cifar10',
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'],
    },
    'MNIST': {
        'num_classes': 10,
        'image_size': 32,
        'channels': 3,
        'mean': (0.1307, 0.1307, 0.1307),
        'std': (0.3081, 0.3081, 0.3081),
        'data_dir': './datasets/MNIST/',
        'weight_dir': './models/MNIST',
        'weight_tag': 'mnist',
        'class_names': [str(i) for i in range(10)],
    },
    'Fashion-MNIST': {
        'num_classes': 10,
        'image_size': 32,
        'channels': 3,
        'mean': (0.2860, 0.2860, 0.2860),
        'std': (0.3530, 0.3530, 0.3530),
        'data_dir': './datasets/Fashion/',
        'weight_dir': './models/Fashion-MNIST',
        'weight_tag': 'fashionmnist',
        'class_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    },
    'TinyImageNet': {
        'num_classes': 200,
        'image_size': 64,
        'channels': 3,
        'mean': (0.4802, 0.4481, 0.3975),
        'std': (0.2770, 0.2691, 0.2821),
        'data_dir': './datasets/TinyImageNet/tiny-imagenet-200/',
        'weight_dir': './models/TinyImageNet',
        'weight_tag': 'tinyimagenet',
        'class_names': None,
    },
}

CIFAR10_MEAN = DATASET_CONFIGS['CIFAR10']['mean']
CIFAR10_STD = DATASET_CONFIGS['CIFAR10']['std']


def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Must be one of {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


TRANSLATION_PARAMS = [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]
SCALE_PARAMS = list(np.arange(0.8, 1, 0.05))
ROTATION_PARAMS = list(range(-30, 30))
CONTRAST_PARAMS = [0.8 + 0.2 * k for k in range(7)]
BRIGHTNESS_PARAMS = [10 + 10 * k for k in range(7)]
BLUR_PARAMS = [k + 1 for k in range(10)]


def get_model_transform(dataset_name):
    cfg = get_dataset_config(dataset_name)
    return transforms.Compose([
        transforms.Resize(cfg['image_size']),
        transforms.CenterCrop(cfg['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std']),
    ])


def _build_resnet50(num_classes, image_size):
    model = tv_models.resnet50(pretrained=False)
    if image_size <= 32:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, num_classes)
    return model


def _build_vgg16_bn(num_classes, image_size):
    model = tv_models.vgg16_bn(pretrained=False)
    if image_size <= 32:
        model.avgpool = nn.Identity()
        in_features = 512
    else:
        in_features = 512 * (image_size // 32) ** 2
        model.avgpool = nn.AdaptiveAvgPool2d((image_size // 32, image_size // 32))
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def _build_mobilenet_v2(num_classes, image_size):
    model = tv_models.mobilenet_v2(pretrained=False)
    if image_size <= 32:
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


def _build_resnet34(num_classes, image_size):
    model = tv_models.resnet34(pretrained=False)
    if image_size <= 32:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def _build_densenet121(num_classes, image_size):
    model = tv_models.densenet121(pretrained=False)
    if image_size <= 32:
        model.features.pool0 = nn.Identity()
    model.classifier = nn.Linear(1024, num_classes)
    return model


def _build_densenet169(num_classes, image_size):
    model = tv_models.densenet169(pretrained=False)
    if image_size <= 32:
        model.features.pool0 = nn.Identity()
    model.classifier = nn.Linear(1664, num_classes)
    return model


def _build_resnet18(num_classes, image_size):
    model = tv_models.resnet18(pretrained=False)
    if image_size <= 32:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def _build_vgg13_bn(num_classes, image_size):
    model = tv_models.vgg13_bn(pretrained=False)
    if image_size <= 32:
        model.avgpool = nn.Identity()
        in_features = 512
    else:
        in_features = 512 * (image_size // 32) ** 2
        model.avgpool = nn.AdaptiveAvgPool2d((image_size // 32, image_size // 32))
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def _build_vgg11_bn(num_classes, image_size):
    model = tv_models.vgg11_bn(pretrained=False)
    if image_size <= 32:
        model.avgpool = nn.Identity()
        in_features = 512
    else:
        in_features = 512 * (image_size // 32) ** 2
        model.avgpool = nn.AdaptiveAvgPool2d((image_size // 32, image_size // 32))
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def _build_vgg19_bn(num_classes, image_size):
    model = tv_models.vgg19_bn(pretrained=False)
    if image_size <= 32:
        model.avgpool = nn.Identity()
        in_features = 512
    else:
        in_features = 512 * (image_size // 32) ** 2
        model.avgpool = nn.AdaptiveAvgPool2d((image_size // 32, image_size // 32))
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def _build_resnet101(num_classes, image_size):
    model = tv_models.resnet101(pretrained=False)
    if image_size <= 32:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, num_classes)
    return model


def _build_alexnet(num_classes, image_size):
    model = tv_models.alexnet(pretrained=False)
    if image_size <= 32:
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.features[3] = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        model.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def _build_lenet3(num_classes, image_size):
    class LeNet3(nn.Module):
        def __init__(self, num_classes):
            super(LeNet3, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
            self.fc1 = nn.Linear(6 * 14 * 14, 120)
            self.fc2 = nn.Linear(120, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return LeNet3(num_classes)


def _build_lenet4(num_classes, image_size):
    class LeNet4(nn.Module):
        def __init__(self, num_classes):
            super(LeNet4, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.fc1 = nn.Linear(16 * 5 * 5, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    return LeNet4(num_classes)


def _build_lenet5(num_classes, image_size):
    class LeNet5(nn.Module):
        def __init__(self, num_classes):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return LeNet5(num_classes)


def _build_resnet50_cifar10():
    return _build_resnet50(10, 32)

def _build_vgg16_bn_cifar10():
    return _build_vgg16_bn(10, 32)

def _build_mobilenet_v2_cifar10():
    return _build_mobilenet_v2(10, 32)

_MODEL_BUILDERS = {
    'resnet50': _build_resnet50_cifar10,
    'vgg16_bn': _build_vgg16_bn_cifar10,
    'mobilenet_v2': _build_mobilenet_v2_cifar10,
}

DEFAULT_WEIGHT_DIR = './pretrained_models'


def _load_state_dict_compatible(weight_path, device='cpu'):
    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def build_model(model_name, dataset_name, weight_dir=None, device='cpu'):
    cfg = get_dataset_config(dataset_name)
    nc = cfg['num_classes']
    sz = cfg['image_size']
    tag = cfg['weight_tag']

    if weight_dir is None:
        weight_dir = cfg['weight_dir']

    builders = {
        'lenet3': lambda: _build_lenet3(nc, sz),
        'lenet4': lambda: _build_lenet4(nc, sz),
        'lenet5': lambda: _build_lenet5(nc, sz),
        'resnet18': lambda: _build_resnet18(nc, sz),
        'resnet34': lambda: _build_resnet34(nc, sz),
        'resnet50': lambda: _build_resnet50(nc, sz),
        'resnet101': lambda: _build_resnet101(nc, sz),
        'vgg11_bn': lambda: _build_vgg11_bn(nc, sz),
        'vgg13_bn': lambda: _build_vgg13_bn(nc, sz),
        'vgg16_bn': lambda: _build_vgg16_bn(nc, sz),
        'vgg19_bn': lambda: _build_vgg19_bn(nc, sz),
        'densenet121': lambda: _build_densenet121(nc, sz),
        'densenet169': lambda: _build_densenet169(nc, sz),
        'mobilenet_v2': lambda: _build_mobilenet_v2(nc, sz),
        'alexnet': lambda: _build_alexnet(nc, sz),
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Must be one of {list(builders.keys())}")

    model = builders[model_name]()

    weight_path = os.path.join(weight_dir, f'{model_name}_{tag}_best.pt')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Weight file not found: {weight_path}\n"
            f"Please run: python models/train_all_models.py --dataset {dataset_name}"
        )

    print(f"Loading {model_name} ({dataset_name}) from {weight_path}")
    state_dict = _load_state_dict_compatible(weight_path, device=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def build_cifar10_model(model_name, weight_dir=DEFAULT_WEIGHT_DIR, device='cpu'):
    return build_model(model_name, 'CIFAR10', weight_dir=weight_dir, device=device)

