
import torch
from torch.utils.data import Dataset
import random
import os
import torch.utils.data.dataloader as DataLoader
import scipy.io as scio
import numpy as np
from scipy.signal import welch


def wgn(x, snr):
    # batch_size, len_x = x.shape
    len_x = 4096
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return x + noise


# 特别注意 本数据集中0代表正样本  1代表负样本
class subDataset(Dataset):
    def __init__(self, datapath, transform, split, allowed_classes=None):
        self.base_path = datapath
        self.split = split

        # 获取所有辐射源路径（全局类别集合）
        split_path = os.path.join(self.base_path, split)
        if os.path.exists(split_path):
            class_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            # 过滤掉非数字目录名，并按数字排序（全局类别ID）
            all_numeric_dirs = []
            for d in class_dirs:
                try:
                    all_numeric_dirs.append(int(d))
                except ValueError:
                    continue
            all_numeric_dirs.sort()

            # 选择本数据集实例实际使用的类别（可为子集），但标签保留为全局ID
            if allowed_classes is not None:
                selected_dirs = [d for d in all_numeric_dirs if d in allowed_classes]
            else:
                selected_dirs = all_numeric_dirs

            # 记录全局与选择后的类别信息
            self.global_num_classes = len(all_numeric_dirs)
            self.class_ids = selected_dirs  # 保留原始全局ID

            # 使用实际的目录名创建路径
            self.emitter_paths = [os.path.join(self.base_path, split, str(i).zfill(2)) for i in self.class_ids]

            # 对上层暴露的类别数：保持为全局类别数量，便于统一模型头部大小
            self.num_classes = self.global_num_classes
        else:
            # 路径不存在时的降级默认（例如LoRa 25类场景）
            self.global_num_classes = 25
            self.class_ids = list(range(self.global_num_classes))
            self.emitter_paths = [os.path.join(self.base_path, split, str(i).zfill(2)) for i in self.class_ids]
            self.num_classes = self.global_num_classes

        self.transform = transform

        self.file_path_label = self._get_file_path_label()

    def _get_file_path_label(self):
        file_path_label = []
        for i, emitter_path in enumerate(self.emitter_paths):
            file_names = os.listdir(emitter_path)
            # 使用全局类别ID作为标签，避免不同客户端之间的重映射不一致
            label = self.class_ids[i]
            file_names = [(os.path.join(emitter_path, file_name), label) for file_name in file_names]
            file_path_label.extend(file_names)
        random.shuffle(file_path_label)
        return file_path_label

    def __len__(self):
        return len(self.file_path_label)

    def __getitem__(self, idx):
        data_path, label = self.file_path_label[idx]
        data = scio.loadmat(data_path)
        Data = np.array(data['z']).astype(np.float32)

        Data_real = Data[0]
        Data_imag = Data[1]

        # 将 NumPy 数组转换为 PyTorch 张量
        Data_real = torch.from_numpy(Data_real)
        Data_imag = torch.from_numpy(Data_imag)

        out = torch.view_as_complex(torch.stack([Data_real, Data_imag], dim=-1))

        return out, label
    '''
    dataset_train = subDataset(datapath='/home/ch/ch/data/',split='train',transform=None )
    #print(dataset)
    print('dataset大小为：', dataset_train.__len__())
    #print(dataset.__getitem__(0))
    #print(dataset[0][0])
    dataset_test = subDataset(datapath='/home/ch/ch/data/',split='Test',transform = None)
    print('dataset大小为：', dataset_test.__len__())
    # 创建DataLoader迭代器
    dataloader_train = DataLoader.DataLoader(dataset_train, batch_size=2, shuffle=False, num_workers=8)
    dataloader_test  = DataLoader.DataLoader(dataset_test,  batch_size=2, shuffle=False, num_workers=8)
    for i, item in enumerate(dataloader_train):
        # print('i:', i)
        data, label = item
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        #print('data:',  data)
        #print('label:', label)
    for i, item in enumerate(dataloader_test):
        # print('i:', i)
        data, label = item
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        #print('data:',  data)
        #print('label:', label)

'''
