import os
import random
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import constants
from shared_config import DATASET_CONFIGS, get_dataset_config

class CIFAR10Dataset(Dataset):
    def __init__(self,
                 args,
                 image_dir=constants.CIFAR10_JPEG_DIR,
                 split='train'):
        super(CIFAR10Dataset).__init__()
        assert split in ['train', 'test']
        self.total_class_num = 10
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        self.image_list = []
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        label = self.class_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label

class MNISTDataset(Dataset):
    def __init__(self, args, image_dir=None, split='train'):
        super(MNISTDataset).__init__()
        assert split in ['train', 'test']
        self.total_class_num = 10
        self.args = args
        cfg = get_dataset_config('MNIST')

        if image_dir is None:
            image_dir = cfg['data_dir']
        self.image_dir = image_dir + split + ('/' if len(split) else '')

        self.transform = transforms.Compose([
            transforms.Resize(cfg['image_size']),
            transforms.Grayscale(num_output_channels=3),
            transforms.CenterCrop(cfg['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])

        self.image_list = []
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        label = self.class_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        return image, label


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = 1

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

def get_loader(args):
    assert args.dataset in ['CIFAR10', 'MNIST', 'ImageNet']
    if args.dataset == 'CIFAR10':
        train_data = CIFAR10Dataset(args, split='train')
        test_data = CIFAR10Dataset(args, split='test')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 10
    elif args.dataset == 'MNIST':
        train_data = MNISTDataset(args, split='train')
        test_data = MNISTDataset(args, split='test')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 10
    elif args.dataset == 'ImageNet':
        train_data = ImageNetDataset(args, split='train')
        test_data = ImageNetDataset(args, split='val')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 1000
    return TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader

class FuzzDataset:
    def __init__(self):
        raise NotImplementedError

    def label2index(self):
        raise NotImplementedError

    def get_len(self):
        return len(self.image_list)

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        index = self.label2index(label)
        assert int(index) < self.args.num_class
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = self.norm(image)
        return (image, index)

    def build(self):
        image_list = []
        label_list = []
        for i in tqdm(range(self.get_len())):
            (image, label) = self.get_item(i)
            image_list.append(image)
            label_list.append(label)
        return image_list, label_list

    def to_numpy(self, image_list, is_image=True):
        image_numpy_list = []
        for i in tqdm(range(len(image_list))):
            image = image_list[i]
            if is_image:
                image_numpy = image.transpose(0, 2).numpy()
            else:
                image_numpy = image.numpy()
            image_numpy_list.append(image_numpy)
        print('Numpy: %d' % len(image_numpy_list))
        return image_numpy_list

    def to_batch(self, data_list, is_image=True):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.args.batch_size == 0:
                batch_list.append(torch.stack(batch, 0))
                batch = []
            batch.append(self.norm(data) if is_image else data)
        if len(batch):
            batch_list.append(torch.stack(batch, 0))
        print('Batch: %d' % len(batch_list))
        return batch_list

class CIFAR10FuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir=constants.CIFAR10_JPEG_DIR,
                 split='test'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                ])
        self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.image_list = []
        
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.class_list.index(label_name)

class MNISTFuzzDataset(FuzzDataset):
    def __init__(self, args, image_dir=None, split='test'):
        cfg = get_dataset_config('MNIST')
        self.args = args
        if image_dir is None:
            image_dir = cfg['data_dir']
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(cfg['image_size']),
            transforms.Grayscale(num_output_channels=3),
            transforms.CenterCrop(cfg['image_size']),
            transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize(cfg['mean'], cfg['std'])
        self.image_list = []

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        index = self.label2index(label)
        assert int(index) < self.args.num_class
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        image = self.norm(image)
        return (image, index)

    def label2index(self, label_name):
        return self.class_list.index(label_name)

class FashionMNISTFuzzDataset(FuzzDataset):
    def __init__(self, args, image_dir=None, split='test'):
        cfg = get_dataset_config('Fashion-MNIST')
        self.args = args
        if image_dir is None:
            image_dir = cfg['data_dir']
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(cfg['image_size']),
            transforms.Grayscale(num_output_channels=3),
            transforms.CenterCrop(cfg['image_size']),
            transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize(cfg['mean'], cfg['std'])
        self.image_list = []

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        index = self.label2index(label)
        assert int(index) < self.args.num_class
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        image = self.norm(image)
        return (image, index)

    def label2index(self, label_name):
        return self.class_list.index(label_name)


class ImageNetFuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir=constants.IMAGENET_JPEG_DIR,
                 label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
                 split='val'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                ])
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []
        
        with open(label2index_file, 'r') as f:
            self.label2index_dict = json.load(f)

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.label2index_dict[label_name]

if __name__ == '__main__':
    pass