import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PathPointDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])

        xyz = torch.from_numpy(data['xyz']).float()        # [N, 3]
        feat = torch.from_numpy(data['feat']).float()      # [N, 3]  ← 你现在是 6=3+3
        target = torch.from_numpy(data['target']).float()  # [N]

        return xyz, feat, target


def collate_fn(batch):
    """
    返回:
        points:  [B, N, 6]
        targets: [B, N, 1]
    """
    B = len(batch)
    N_max = max(xyz.shape[0] for xyz, _, _ in batch)

    points_list = []
    targets_list = []

    for xyz, feat, target in batch:
        N = xyz.shape[0]
        pad_n = N_max - N

        xyz_pad = torch.cat(
            [xyz, torch.zeros(pad_n, 3)], dim=0
        )
        feat_pad = torch.cat(
            [feat, torch.zeros(pad_n, feat.shape[1])], dim=0
        )

        points = torch.cat([xyz_pad, feat_pad], dim=1)  # [N, 6]
        points_list.append(points)

        targets_list.append(
            torch.cat([target, torch.zeros(pad_n)], dim=0)
        )

    points = torch.stack(points_list)          # [B, N, 6]
    targets = torch.stack(targets_list).unsqueeze(-1)  # [B, N, 1]

    return points, targets


def build_dataloader(
    data_files,
    batch_size=8,
    shuffle=True,
    num_workers=4
):
    dataset = PathPointDataset(data_files)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
