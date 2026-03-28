"""
RATR Dataset Reader
Dataset: proj_test_dataset_pkl - 3 radar target classes (E2D / P3C / P8A)
Signal format: 1D sequence with 1024 samples (per sample)

Each .pkl file structure:
  {
    'data': np.ndarray shape (1024, N),
    'var': str,
    'source': str
  }
Where each column of 'data' is one sample.
"""

import os
import sys
import pickle
import time
import bisect
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


_RATR_FILE_CACHE = {}


def _ensure_numpy_compat_for_old_pickles():
    """Fix common pickle compatibility issues where pickles reference numpy._core.*."""
    try:
        import numpy.core as _np_core

        sys.modules.setdefault("numpy._core", _np_core)
        sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
        sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
    except Exception:
        pass


def _infer_class_from_filename(stem: str) -> str:
    s = stem.upper()
    if s.startswith("E2D"):
        return "E2D"
    if s.startswith("P3C"):
        return "P3C"
    if s.startswith("P8A"):
        return "P8A"
    raise ValueError(f"Cannot infer class from filename stem: {stem}")


class RATRDataset(Dataset):
    """RATR dataset loader (proj_test_dataset_pkl) with caching and stratified split.

    It does NOT concatenate all samples into one huge array (to reduce RAM).
    Instead it builds global indices across files and loads each file array on demand.

    Args:
        data_dir: folder containing *.pkl files
        split: 'train' | 'val' | 'test'
        seed: random seed
        add_noise: whether to add noise to the 1D real signal
        noise_type: 'awgn' | 'factor'
        noise_snr_db: AWGN SNR in dB
        noise_factor: factor noise scale (noise_power = noise_factor * signal_power)
    """

    def __init__(
        self,
        data_dir,
        split="train",
        seed=42,
        add_noise=False,
        noise_type="awgn",
        noise_snr_db=15,
        noise_factor=0.1,
    ):
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_snr_db = noise_snr_db
        self.noise_factor = noise_factor

        self.class_names = ["E2D", "P3C", "P8A"]
        self.class_to_label = {c: i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        _ensure_numpy_compat_for_old_pickles()

        self._init_index()

    def _list_pkl_files(self):
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pkl")]
        files.sort()
        if not files:
            raise FileNotFoundError(f"No .pkl files found in: {self.data_dir}")
        return files

    def _load_file_data(self, file_path: str):
        """Load one pkl and return its data array as float32 np.ndarray with shape (1024, N)."""
        if file_path in _RATR_FILE_CACHE:
            return _RATR_FILE_CACHE[file_path]

        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict) or "data" not in obj:
            raise ValueError(f"Unexpected pkl structure in {file_path}: expected dict with key 'data'")

        data = obj["data"]
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.ndim != 2 or data.shape[0] != 1024:
            raise ValueError(f"Unexpected data shape in {file_path}: {data.shape} (expected (1024, N))")

        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)

        _RATR_FILE_CACHE[file_path] = data
        return data

    def _init_index(self):
        """Build global index across files and create stratified split indices."""
        t0 = time.time()

        files = self._list_pkl_files()
        self.file_paths = [os.path.join(self.data_dir, f) for f in files]
        self.file_stems = [os.path.splitext(f)[0] for f in files]

        file_counts = []
        file_labels = []
        class_to_global_indices = defaultdict(list)

        offsets = [0]
        total = 0

        print(f"正在初始化 RATR 数据集索引: {self.data_dir}")

        for fp, stem in zip(self.file_paths, self.file_stems):
            cls_name = _infer_class_from_filename(stem)
            label = self.class_to_label[cls_name]

            data = self._load_file_data(fp)
            n = int(data.shape[1])

            file_counts.append(n)
            file_labels.append(label)

            start = total
            end = total + n
            class_to_global_indices[label].extend(range(start, end))

            total = end
            offsets.append(total)

        self.file_counts = file_counts
        self.file_labels = file_labels
        self.file_offsets = offsets
        self.total_samples = total

        rng = np.random.RandomState(self.seed)

        train_idx = []
        val_idx = []
        test_idx = []

        for label in range(self.num_classes):
            indices = np.array(class_to_global_indices.get(label, []), dtype=np.int64)
            if indices.size == 0:
                continue
            rng.shuffle(indices)

            n = int(indices.size)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)

            train_idx.append(indices[:n_train])
            val_idx.append(indices[n_train : n_train + n_val])
            test_idx.append(indices[n_train + n_val :])

        self.train_indices = np.concatenate(train_idx) if train_idx else np.empty((0,), dtype=np.int64)
        self.val_indices = np.concatenate(val_idx) if val_idx else np.empty((0,), dtype=np.int64)
        self.test_indices = np.concatenate(test_idx) if test_idx else np.empty((0,), dtype=np.int64)

        rng.shuffle(self.train_indices)
        rng.shuffle(self.val_indices)
        rng.shuffle(self.test_indices)

        if self.split == "train":
            self.indices = self.train_indices
        elif self.split == "val":
            self.indices = self.val_indices
        else:
            self.indices = self.test_indices

        elapsed = time.time() - t0
        print(f"✅ 索引初始化完成 (耗时: {elapsed:.2f}秒)")
        print(f"   类别: {self.class_names} (num_classes={self.num_classes})")
        print(f"   总样本数: {self.total_samples}")
        print(f"   训练集: {len(self.train_indices)}")
        print(f"   验证集: {len(self.val_indices)}")
        print(f"   测试集: {len(self.test_indices)}")
        print(f"   当前 split='{self.split}': {len(self.indices)}")

    def __len__(self):
        return int(self.indices.size)

    def _global_to_file_local(self, gidx: int):
        file_idx = bisect.bisect_right(self.file_offsets, gidx) - 1
        if file_idx < 0 or file_idx >= len(self.file_paths):
            raise IndexError(f"Global index out of range: {gidx}")
        local_idx = gidx - self.file_offsets[file_idx]
        return file_idx, int(local_idx)

    def __getitem__(self, idx):
        gidx = int(self.indices[idx])
        file_idx, local_idx = self._global_to_file_local(gidx)

        fp = self.file_paths[file_idx]
        data = self._load_file_data(fp)

        x = data[:, local_idx]
        y = self.file_labels[file_idx]

        if self.add_noise:
            x = self._add_noise(x)

        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y_tensor = torch.tensor(int(y), dtype=torch.long)
        return x_tensor, y_tensor

    def _add_noise(self, x: np.ndarray) -> np.ndarray:
        if self.noise_type == "awgn":
            return self._add_awgn_noise(x)
        if self.noise_type == "factor":
            return self._add_factor_noise(x)
        raise ValueError(f"Unknown noise type: {self.noise_type}")

    def _add_awgn_noise(self, x: np.ndarray) -> np.ndarray:
        p_signal = float(np.mean(np.square(x)))
        if p_signal <= 0:
            return x
        p_noise = p_signal / (10 ** (self.noise_snr_db / 10))
        noise = np.random.normal(0.0, np.sqrt(p_noise), size=x.shape).astype(np.float32)
        return x + noise

    def _add_factor_noise(self, x: np.ndarray) -> np.ndarray:
        p_signal = float(np.mean(np.square(x)))
        if p_signal <= 0:
            return x
        p_noise = float(self.noise_factor) * p_signal
        noise = np.random.normal(0.0, np.sqrt(p_noise), size=x.shape).astype(np.float32)
        return x + noise


def build_ratr_dataloader(
    data_dir,
    split="train",
    batch_size=32,
    num_workers=0,
    shuffle=None,
    **dataset_kwargs,
):
    from torch.utils.data import DataLoader

    ds = RATRDataset(data_dir=data_dir, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def load_ratr_numpy(data_path, split="test", seed=42, max_samples=None, **dataset_kwargs):
    if os.path.isfile(data_path) and data_path.lower().endswith('.mat'):
        try:
            import scipy.io
        except Exception as e:
            raise ImportError(
                "scipy is required to load .mat files for RATR. Install it via pip/conda.") from e

        mat = scipy.io.loadmat(data_path)
        if 'tarHRRP_inScene_db' not in mat or 'tarLabelIsInsight_inScene' not in mat:
            keys = [k for k in mat.keys() if not k.startswith('__')]
            raise KeyError(
                f"Unexpected RATR .mat keys: {keys}. Required: tarHRRP_inScene_db, tarLabelIsInsight_inScene")

        X_raw = mat['tarHRRP_inScene_db']
        y_raw = mat['tarLabelIsInsight_inScene']

        if not isinstance(X_raw, np.ndarray):
            X_raw = np.asarray(X_raw)
        if not isinstance(y_raw, np.ndarray):
            y_raw = np.asarray(y_raw)

        if X_raw.ndim != 2 or X_raw.shape[0] != 1024:
            raise ValueError(
                f"Unexpected tarHRRP_inScene_db shape: {X_raw.shape} (expected (1024, N))")

        y_flat = y_raw.reshape(-1).astype(np.int64, copy=False)
        if y_flat.size != X_raw.shape[1]:
            raise ValueError(
                f"Label size mismatch: X has N={X_raw.shape[1]} but y has {y_flat.size}")

        y_min = int(y_flat.min())
        y_max = int(y_flat.max())
        if y_min == 1 and y_max == 3:
            y_flat = y_flat - 1
        elif y_min == 0 and y_max in (1, 2):
            pass
        else:
            raise ValueError(
                f"Unexpected label range for RATR .mat: min={y_min}, max={y_max} (expected 0..2 or 1..3)")

        # (1024, N) -> (N, 1, 1024)
        X = X_raw.T.astype(np.float32, copy=False)[:, None, :]
        y = y_flat

        # Deterministic split: test_size=0.2, val_size=0.1 from remaining
        n = X.shape[0]
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)

        test_size = 0.2
        val_size = 0.1

        n_test = int(n * test_size)
        n_test = max(1, min(n - 2, n_test))

        rem = n - n_test
        n_val = int(rem * val_size)
        n_val = max(1, min(rem - 1, n_val))

        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]

        split = (split or 'test').lower()
        if split == 'train':
            X, y = X[train_idx], y[train_idx]
        elif split in ('val', 'valid', 'validation'):
            X, y = X[val_idx], y[val_idx]
        elif split == 'test':
            X, y = X[test_idx], y[test_idx]
        else:
            raise ValueError(f"Unsupported split: {split} (expected 'train'|'val'|'test')")

        if max_samples is not None:
            X = X[: int(max_samples)]
            y = y[: int(max_samples)]

        return X, y

    if os.path.isdir(data_path):
        ds = RATRDataset(data_dir=data_path, split=split, seed=seed, **dataset_kwargs)
        n = len(ds)
        if max_samples is not None:
            n = min(n, int(max_samples))

        X = np.empty((n, 1, 1024), dtype=np.float32)
        y = np.empty((n,), dtype=np.int64)
        for i in range(n):
            xi, yi = ds[i]
            X[i, 0, :] = xi.numpy()
            y[i] = int(yi.item())
        return X, y

    _ensure_numpy_compat_for_old_pickles()
    stem = os.path.splitext(os.path.basename(data_path))[0]
    label = {"E2D": 0, "P3C": 1, "P8A": 2}[_infer_class_from_filename(stem)]

    with open(data_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "data" not in obj:
        raise ValueError(f"Unexpected pkl structure in {data_path}: expected dict with key 'data'")
    data = obj["data"]
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.ndim != 2 or data.shape[0] != 1024:
        raise ValueError(f"Unexpected data shape in {data_path}: {data.shape} (expected (1024, N))")
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)

    X = data.T[:, None, :]
    y = np.full((X.shape[0],), label, dtype=np.int64)

    if max_samples is not None:
        X = X[: int(max_samples)]
        y = y[: int(max_samples)]

    return X, y
