from torchvision import transforms
import ssl
import torchvision
from torch.utils.data import DataLoader


class Data:
    """得到训练数据"""
    def __init__(self):
        self.transformers_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transformers_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_batch_size = 50
        self.test_batch_size = 10
        self.train_data = torchvision.datasets.CIFAR10(root=r'D:\\download\\cifar10',
                                                       train=True,
                                                       download=True,
                                                       transform=self.transformers_train)

        self.test_data = torchvision.datasets.CIFAR10(root=r'D:\\download\\cifar10',
                                                      train=False,
                                                      download=True,
                                                      transform=self.transformers_test)

        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       num_workers=0)

        self.test_loader = DataLoader(dataset=self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      num_workers=0)
