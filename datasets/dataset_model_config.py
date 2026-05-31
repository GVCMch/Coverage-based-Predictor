DATASET_MODELS = {
    'MNIST': ['lenet3', 'lenet4', 'lenet5'],
    'Fashion-MNIST': ['vgg11_bn', 'vgg13_bn', 'resnet18', 'resnet34', 'alexnet', 'mobilenet_v2'],
    'CIFAR10': ['vgg13_bn', 'vgg16_bn', 'resnet34', 'resnet50', 'densenet121', 'mobilenet_v2'],
    'TinyImageNet': ['vgg19_bn', 'resnet34', 'resnet50', 'densenet121', 'densenet169'],
}

MODEL_NAME_TO_ID = {
    'MNIST': {
        'lenet3': 0,
        'lenet4': 1,
        'lenet5': 2,
    },
    'Fashion-MNIST': {
        'vgg11_bn': 0,
        'vgg13_bn': 1,
        'resnet18': 2,
        'resnet34': 3,
        'alexnet': 4,
        'mobilenet_v2': 5,
    },
    'CIFAR10': {
        'vgg13_bn': 0,
        'vgg16_bn': 1,
        'resnet34': 2,
        'resnet50': 3,
        'densenet121': 4,
        'mobilenet_v2': 5,
    },
    'TinyImageNet': {
        'vgg19_bn': 0,
        'resnet34': 1,
        'resnet50': 2,
        'densenet121': 3,
        'densenet169': 4,
    },
}

def get_models_for_dataset(dataset_name):
    return DATASET_MODELS.get(dataset_name, [])

def get_model_id(dataset_name, model_name):
    return MODEL_NAME_TO_ID.get(dataset_name, {}).get(model_name, 0)

def get_num_models(dataset_name):
    return len(DATASET_MODELS.get(dataset_name, []))
