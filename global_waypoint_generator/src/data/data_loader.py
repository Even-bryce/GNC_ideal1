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
        
        # points: [N, D], target: [N, 1], waypoints: [M, 3]
        points = torch.from_numpy(data['points']).float() 
        target = torch.from_numpy(data['labels']).float()
        waypoints = torch.from_numpy(data['waypoints']).float()

        # if target.dim() > 1 and target.shape[-1] == 2:
        #     target = target[:, 0:1]

        return points, target, waypoints

def collate_fn(batch):
    """
    修复后的 collate_fn：
    1. 动态计算当前 Batch 的最大长度 N_max
    2. 生成 points_batch, targets_batch
    3. 生成 mask_batch (True表示真点，False表示padding)
    """
    # 获取当前 batch 中最大的点数
    N_max = max(item[0].shape[0] for item in batch)
    B = len(batch)
    D = batch[0][0].shape[1] # 特征维度

    # 预先分配内存（比不断 cat 效率更高）
    # points_batch 初始化为 0
    points_batch = torch.zeros(B, N_max, D)
    # targets_batch 初始化为 0 或 -1 (取决于你的 Loss 处理)
    # targets_batch = torch.zeros(B, N_max, 1)
    targets_batch = torch.zeros(B, N_max, 2)
    # mask_batch 初始化为 False (或 0)
    mask_batch = torch.zeros(B, N_max, dtype=torch.bool)

    waypoints_list = []

    for i, (points, target, waypoints) in enumerate(batch):
        N = points.shape[0]
        
        # 填充数据
        points_batch[i, :N, :] = points
        targets_batch[i, :N, :] = target
        
        # 设置有效区域的掩码为 True
        mask_batch[i, :N] = True
        
        # 变长航路点依然保留在 list 中
        waypoints_list.append(waypoints)

    return points_batch, targets_batch, mask_batch, waypoints_list

def build_dataloader(data_files, batch_size=8, shuffle=True, num_workers=0):
    dataset = PathPointDataset(data_files)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True # 开启以加速 CPU 到 GPU 的数据传输
    )