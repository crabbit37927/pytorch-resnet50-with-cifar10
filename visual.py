import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import random


class Visual:
    def __init__(self, model):
        self.model = model
        self.data = model.data

    def visual_train(self):
        """可视化训练过程中模型损失值和准确率的变化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # 绘制准确率图
        ax1.plot(range(1, len(self.model.acc_record) + 1), self.model.acc_record,
                 marker='o', color='orange', label='accuracy')
        for i, acc in enumerate(self.model.acc_record):
            ax1.text(i + 1, acc, f'{acc:.2f}', fontsize=8, ha='center', va='bottom')
        ax1.set_xticks(range(1, len(self.model.acc_record) + 1))
        ax1.set_xlabel('Epoch')
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_ylim([0, 1])
        ax1.legend()

        # 绘制损失值图
        ax2.plot(range(1, len(self.model.loss_record) + 1), self.model.loss_record, marker='o', label='loss')
        for i, loss in enumerate(self.model.loss_record):
            ax2.text(i + 1, loss, f'{loss:.2f}', fontsize=8, ha='center', va='top')
        ax2.set_xticks(range(1, len(self.model.loss_record) + 1))
        ax2.set_xlabel('Epoch')
        ax2.legend()

        plt.show()

    def draw_matrix(self):
        """绘制测试集的混淆矩阵"""
        class_names = [str(i) for i in range(10)]
        self._plot_confusion_matrix(class_names)

    def _plot_confusion_matrix(self, classes):
        """绘制混淆矩阵的方法"""
        plt.figure(figsize=(12, 10))
        flipped_conf_matrix = np.flipud(self.model.conf_matrix)
        plt.imshow(flipped_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, reversed(classes))

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(flipped_conf_matrix[i, j]), ha="center", va="center", color="black")

        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.show()

    def visual_test(self):
        """选择10张图片并对预测结果进行可视化"""
        dict_class = {}
        for i in range(len(self.model.data.classes)):
            dict_class[i] = self.model.data.classes[i]

        # 进行图片的反归一化
        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage()
        ])

        images_to_show = []
        labels_to_show = []
        predictions = []
        for idx, (image, label) in enumerate(self.model.data.test_data):
            if idx < 10:
                image = image.to('cuda')
                original_image = reverse_transform(image)
                images_to_show.append(original_image)
                labels_to_show.append(label)

                # 获取图片的预测结果
                y = torch.argmax(F.softmax(self.model.net(image.unsqueeze(0)), dim=1))
                y = str(y)
                y = int(y[7])
                predictions.append(y)

        # 将十张图片绘制在一张图上
        plt.figure(figsize=(16, 16))
        for i in range(len(images_to_show)):
            plt.subplot(2, 5, i + 1)
            plt.title(dict_class[predictions[i]],
                      color='green' if self.model.data.classes[labels_to_show[i]] == dict_class[predictions[i]]
                      else 'red'
                      )
            plt.imshow(images_to_show[i])
            plt.axis('off')
        plt.show()

    def show_feature(self):
        """显示特征图"""
        # 加载已经训练好的模型
        model = torchvision.models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10)
        model.load_state_dict(torch.load('alexnet_fine_tuned.pth'))
        feature_extractor = create_feature_extractor(model, return_nodes={"conv1": "conv1", "layer1": "layer1",
                                                                          "layer2": "layer2", "layer3": "layer3",
                                                                          "layer4": "layer4", "fc": "fc"})

        # 将图像反归一化
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage()
        ])

        # 提取出测试集内第一张图片
        for images, labels in self.data.test_loader:
            first_image = images[0]
            label = labels[0]
            break

        # 展示原图
        transformed_image = transform(first_image)
        plt.imshow(transformed_image)
        plt.title(f'Original Image - Label: {label}')
        plt.axis('off')
        plt.show()

        # 将图像类型转换回张量
        transform_tensor = transforms.Compose([
            transforms.ToTensor()
        ])

        img = transform_tensor(transformed_image).unsqueeze(0)
        out = feature_extractor(img)
        conv1_output = out["conv1"]
        layer1_output = out["layer1"]
        layer2_output = out["layer2"]
        layer3_output = out["layer3"]
        layer4_output = out["layer4"]
        self._draw_features(conv1_output)
        self._draw_features(layer1_output)
        self._draw_features(layer2_output)
        self._draw_features(layer3_output)
        self._draw_features(layer4_output)

    def _draw_features(self, type_output):
        """随机选择25张特征图进行展示"""
        type_output = type_output.data.squeeze(0)
        channel_num = self._random_num(size=25, end=type_output.shape[0])
        plt.figure(figsize=(10, 10))
        for index, channel in enumerate(channel_num):
            ax = plt.subplot(5, 5, index + 1, )
            plt.imshow(type_output[channel, :, :])
        plt.show()

    def _random_num(self, size, end):
        """获取指定个数的随机数"""
        range_ls = [i for i in range(end)]
        num_ls = []
        for i in range(size):
            num = random.choice(range_ls)
            range_ls.remove(num)
            num_ls.append(num)
        return num_ls
