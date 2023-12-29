import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import torchvision

import time
from time import sleep

from sklearn.metrics import confusion_matrix, classification_report


class Model(nn.Module):
    """创建并训练模型，将训练结果可视化"""
    def __init__(self, data):
        super(Model, self).__init__()
        self.num_classes = 10
        self.epoch = 10
        self.acc_record = []
        self.loss_record = []

        self.data = data()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader
        self.train_batch_size = self.data.train_batch_size
        self.test_batch_size = self.data.test_batch_size

        # 定义损失函数，模型，梯度更新方法
        self.criterion = torch.nn.CrossEntropyLoss()
        self._get_net()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        self.lr_step = StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.name = ''
        self.conf_matrix = None

    def _get_net(self):
        """导入预训练过的模型"""
        # 选择要进行训练的模型
        self.name = input("请输入你要选择的模型名称(resnet/alexnet/Densenet)：")
        self.name = self.name.lower()

        if self.name == 'resnet':
            resnet = torchvision.models.resnet50(pretrained=True)
            in_features = resnet.fc.in_features
            resnet.fc = nn.Linear(in_features, self.num_classes)
            resnet = resnet.cuda()
            self.net = resnet
            self.path = "resnet_fine_tuned.pth"
        elif self.name == 'alexnet':
            alexnet = torchvision.models.alexnet(pretrained=True)
            in_features = alexnet.classifier[-1].in_features
            alexnet.classifier[-1] = nn.Linear(in_features, self.num_classes)
            alexnet = alexnet.cuda()
            self.net = alexnet
            self.path = "alexnet_fine_tuned.pth"
        elif self.name == 'densenet':
            densenet = torchvision.models.DenseNet()
            densenet = densenet.cuda()
            self.net = densenet
            self.path = "densenet_fine_tuned_test.pth"

    def train_model(self):
        """进行模型的训练"""
        start_time = time.time()
        for epoch in range(self.epoch):
            batch = 0
            acc_epoch = 0
            loss_epoch = 0

            for inputs, labels in self.train_loader:
                labels = labels.to('cuda')
                inputs = inputs.to('cuda')

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                torch.cuda.empty_cache()

                batch += 1
                probabilities = F.softmax(outputs, dim=1)
                acc = float(torch.sum(labels == torch.argmax(probabilities, 1))) / self.train_batch_size
                loss = round(float(loss), 4)
                print(f"epoch: {epoch + 1}, batch: {batch}, loss: {loss}, acc: {acc}")

                acc_epoch += acc
                loss_epoch += loss
            self.lr_step.step()

            acc_epoch = round(acc_epoch / batch, 2)
            loss_epoch = round(loss_epoch / batch, 2)
            self.acc_record.append(acc_epoch)
            self.loss_record.append(loss_epoch)
            print(f'epoch: {epoch + 1},loss_average: {loss_epoch}, acc_average: {acc_epoch}')
            print("—————————————————————————————————————————————————————————————————————————")
            sleep(3)

        end_time = time.time()
        run_time = round(end_time - start_time)
        hour = run_time // 3600
        minute = (run_time - 3600 * hour) // 60
        second = run_time - 3600 * hour - 60 * minute
        print(f'该模型训练时间：{hour}小时{minute}分钟{second}秒，接下来进行模型的测试')
        print("Saving model to:", self.path)
        torch.save(self.net.state_dict(), self.path)

    def test_model(self):
        """进行模型的测试"""
        self.net.load_state_dict(torch.load(self.path))
        self.net.eval()
        accs = 0
        all_labels = []
        all_predictions = []
        batch = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                output = self.net(inputs)

                probabilities = F.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

                accuracy = float(torch.sum(labels == torch.argmax(probabilities, 1))) / self.test_batch_size
                batch += 1
                accs += accuracy

        accs = accs / batch
        print(f"测试集的精确值为: {accs}")

        # 计算混淆矩阵和分类报告
        self.conf_matrix = confusion_matrix(all_labels, all_predictions)
        print("混淆矩阵:")
        print(self.conf_matrix)

        class_report = classification_report(all_labels, all_predictions)
        print("分类报告:")
        print(class_report)
