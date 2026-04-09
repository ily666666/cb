import os
import json
import pickle
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import h5py
from torch.utils.data import Dataset
from scipy.io import loadmat

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

RATR_RESULT_DIR = r"/home/lmc501/damawa/moxingyasuo/proj/proj_RATR_complex/ratr_raw/result"
RATR_DATASET_DIR = r"/home/lmc501/damawa/moxingyasuo/proj/proj_RATR_complex/dataset"

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x / keep_prob * random_tensor


class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

def conv1x3_1d(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


class SEBottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, reduction=16, downsample=None, drop_path_prob=0.0, dropout1d_prob=0.2):
        super(SEBottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.se = SELayer1D(planes * self.expansion, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out = self.drop_path(out)
        out += identity
        out = self.relu(out)
        return out


class SEResNet1D(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=3,
        reduction=16,
        in_channels=1,
        dropout_rate=0.3,
        drop_path_rate=0.2
    ):
        super(SEResNet1D, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


        self.total_blocks = int(sum(layers))
        self._block_idx = 0
        self.drop_path_rate = float(drop_path_rate)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)


        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _get_drop_path_prob(self):
        if self.total_blocks <= 1:
            return self.drop_path_rate
        return self.drop_path_rate * (self._block_idx / (self.total_blocks - 1))

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        dp = self._get_drop_path_prob()
        layers.append(block(
            self.inplanes, planes,
            stride=stride, reduction=reduction, downsample=downsample,
            drop_path_prob=dp
        ))
        self._block_idx += 1

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            dp = self._get_drop_path_prob()
            layers.append(block(
                self.inplanes, planes,
                stride=1, reduction=reduction, downsample=None,
                drop_path_prob=dp
            ))
            self._block_idx += 1

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def se_resnet101_1d(**kwargs):
    return SEResNet1D(SEBottleneck1D, [3, 4, 23, 3], **kwargs)


def _load_mat_array(mat_file: str, var_name: str):

    try:
        with h5py.File(mat_file, "r") as f:
            if var_name in f:
                arr = np.array(f[var_name])
            else:
                keys = [k for k in f.keys() if not k.startswith("#")]
                if not keys:
                    raise KeyError("No valid key in HDF5 mat")
                arr = np.array(f[keys[0]])
            return np.array(arr)
    except Exception:
        mat = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        if var_name in mat:
            arr = mat[var_name]
        else:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            if not keys:
                raise ValueError(f"{mat_file} 中未找到有效变量")
            arr = mat[keys[0]]
        arr = arr.transpose(1, 0)
        return np.array(arr)

class MatSignalTestDataset(Dataset):

    def __init__(self, mat_files, labels, var_name="a"):
        assert len(mat_files) == len(labels), "mat_files 和 labels 长度必须一致"

        self.mat_files = mat_files
        self.labels = labels
        self.var_name = var_name

        self.data = []
        self.targets = []
        self.file_offsets = [0]
        self.file_meta = []

        for mat_file, label in zip(mat_files, labels):
            arr = _load_mat_array(mat_file, var_name=var_name)
            arr = np.asarray(arr)

            if arr.ndim == 4:
                arr = arr.transpose(3, 2, 1, 0)
                El, Az, Sets, L = arr.shape
                d2 = arr.reshape(El * Az * Sets, L).astype(np.float32)
                mode = "4d"

            elif arr.ndim == 3:
                El, Az, L = arr.shape
                Sets = 1
                d2 = arr.reshape(El * Az, L).astype(np.float32)
                mode = "3d"

            elif arr.ndim == 2:
                N, L = arr.shape
                El, Az, Sets = -1, -1, -1
                d2 = arr.astype(np.float32)
                mode = "2d"

            else:
                raise ValueError(
                    f"[MatSignalTestDataset]不支持该格式的数据 shape={arr.shape}"
                )

            Ni = int(d2.shape[0])

            self.data.append(d2)
            self.targets.append(np.full((Ni,), int(label), dtype=np.int64))

            self.file_meta.append({
                "path": mat_file,
                "label": int(label),
                "mode": mode,
                "El": int(El),
                "Az": int(Az),
                "Sets": int(Sets),
                "L": int(L),
                "N": int(Ni),
            })

            self.file_offsets.append(self.file_offsets[-1] + Ni)

        self.total = int(self.file_offsets[-1])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        file_id = int(np.searchsorted(self.file_offsets, idx, side="right") - 1)
        local_idx = int(idx - self.file_offsets[file_id])

        d2 = self.data[file_id]
        y = int(self.targets[file_id][local_idx])

        meta_f = self.file_meta[file_id]
        mode = meta_f["mode"]

        if mode == "4d":
            Az = meta_f["Az"]
            Sets = meta_f["Sets"]

            ele_idx = local_idx // (Az * Sets)
            rem = local_idx % (Az * Sets)
            az_idx = rem // Sets
            set_idx = rem % Sets

        elif mode == "3d":
            Az = meta_f["Az"]

            ele_idx = local_idx // Az
            az_idx = local_idx % Az
            set_idx = -1

        else:
            ele_idx = -1
            az_idx = -1
            set_idx = -1

        x = torch.from_numpy(d2[local_idx]).float().unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        meta = {
            "global_idx": int(idx),
            "file_id": int(file_id),
            "local_idx": int(local_idx),
            "ele_idx": int(ele_idx),
            "az_idx": int(az_idx),
            "set_idx": int(set_idx),
            "mode": mode,
            "El": int(meta_f["El"]),
            "Az": int(meta_f["Az"]),
            "Sets": int(meta_f["Sets"]),
        }

        return x, y, meta

def build_model(model_name: str, num_classes: int, in_channels: int = 1, dropout_rate: float = 0.3):
    name = model_name.lower()
    if name == "se_resnet101_1d":
        return se_resnet101_1d(num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"未知模型名称: {model_name}")


def safe_load_weights(model, model_path: str, device: torch.device):

    ckpt = torch.load(model_path, map_location=device)

    state = None
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "clientA_model", "model", "state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break

    if state is None and isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        if keys and isinstance(ckpt[keys[0]], torch.Tensor):
            state = ckpt

    if state is None:
        raise RuntimeError(f"无法从 checkpoint 解析权重：{model_path}")

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"[safe_load_weights] missing={len(missing)}, unexpected={len(unexpected)} (strict=True)")


def load_dataset_from_hparams(hp: Dict[str, Any]):
    if "test_loader" in hp and hp["test_loader"]:
        tl = hp["test_loader"]
        file_name = tl["file_name"]
        parent_folder = tl.get("parent_folder", "")
        p = os.path.join(parent_folder, file_name) if parent_folder else file_name
        with open(p, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    test_files = hp["test_files"]
    labels = hp["labels"]
    var_name = hp.get("var_name", "tarHRRP_inScene_db")
    dataset = MatSignalTestDataset(test_files, labels, var_name=var_name)
    return dataset


# ================== 任务-雷达探测目标识别==================
def ratr_predict_and_analyze(dataset, model, device, batch_size, shuffle, save_dir):

    plot_filename = f"confusion_matrix.png"
    plot_path = os.path.join(save_dir, plot_filename)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    device = torch.device(device)

    def _parse_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], None
            raise RuntimeError(f"Unexpected batch length: {len(batch)}")
        if isinstance(batch, dict):
            return batch["x"], batch["y"], batch.get("meta", None)
        raise RuntimeError(f"Unexpected batch type: {type(batch)}")

    model.eval()
    model.to(device)

    all_preds = []
    all_true = []

    for batch in test_loader:
        X_batch, Y_batch, meta = _parse_batch(batch)
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = Y_batch.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(X_batch)
            _, predicted = torch.max(output, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    all_labels = np.unique(np.concatenate((all_true, all_preds)))
    cm = confusion_matrix(all_true, all_preds, labels=all_labels)


    plt.figure(figsize=(8, 6))
    row_sums = cm.sum(axis=1)
    cm_percentage = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_percentage[i] = cm[i] / row_sums[i]
        else:
            cm_percentage[i] = 0

    class_name = [f'Class 1', f'Class 2', f'Class 3']
    xticklabel = [f'Class 1', f'Class 2', f'Class 3']
    valid_row = row_sums > 0
    cm_percentage = cm_percentage[valid_row, :]
    yticklabel = [class_name[i] for i in range(len(class_name)) if valid_row[i]]
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=xticklabel,
                yticklabels=yticklabel)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return {
        "plot_path": plot_path,
    }


def run(task_id: str, hparams_path: str):
    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)

    save_dir = hp.get("save_dir", RATR_RESULT_DIR).format(task_id)
    os.makedirs(save_dir, exist_ok=True)

    device = hp.get("device", DEVICE)

    model_path = hp['complex_model_path']
    model_name = hp['complex_model_name']
    model = build_model(
        model_name=model_name,
        num_classes=int(hp.get("num_classes", 3)),
        in_channels=int(hp.get("in_channels", 1)),
        dropout_rate=float(hp.get("dropout_rate", 0.3)),
    )

    # 计算参数量
    total_params = sum(model.numel() for model in model.parameters() if model.numel())
    size_bytes= total_params * 4
    size_mb = size_bytes / 1024 / 1024
    print(f"Total parameters: {size_mb}") # 125.9270133972168 KB

    # 加载权重
    safe_load_weights(model, model_path=model_path, device=torch.device(device))
    # 加载数据
    dataset = load_dataset_from_hparams(hp)

    ret = ratr_predict_and_analyze(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=int(hp.get("batch_size", 512)),
        shuffle=bool(hp.get("shuffle", True)),
        save_dir=save_dir,
    )

    print(f"[Done] confusion_matrix: {ret['plot_path']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="ratr_raw_1")
    parser.add_argument("--hparams", type=str, default=r"/home/lmc501/damawa/moxingyasuo/proj/proj_RATR_complex/ratr_raw/input/proj_RATR_complex.json")
    args = parser.parse_args()
    run(task_id=args.task_id, hparams_path=args.hparams)