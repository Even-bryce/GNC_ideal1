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
        
        # 💡 注释更新：points: [N, 9], target: [N, 2] (双通道), waypoints: [M, 3]
        points = torch.from_numpy(data['points']).float() 
        target = torch.from_numpy(data['labels']).float()
        waypoints = torch.from_numpy(data['waypoints']).float()

        return points, target, waypoints

def collate_fn(batch):
    """
    修复后的 collate_fn：支持多通道标签的动态组装
    1. 动态计算当前 Batch 的最大长度 N_max
    2. 动态获取特征维度 D 和 标签维度 L
    3. 生成 points_batch, targets_batch 和 mask_batch
    """
    # 获取当前 batch 中最大的点数
    N_max = max(item[0].shape[0] for item in batch)
    B = len(batch)
    
    # 动态获取特征维度 (D=9) 和 标签维度 (L=2)
    D = batch[0][0].shape[1] 
    L = batch[0][1].shape[1] if len(batch[0][1].shape) > 1 else 1

    # 预先分配内存
    points_batch = torch.zeros(B, N_max, D)
    
    # 💡 核心修改：将写死的 1 改为动态维度 L
    targets_batch = torch.zeros(B, N_max, L) 
    
    mask_batch = torch.zeros(B, N_max, dtype=torch.bool)
    waypoints_list = []

    for i, (points, target, waypoints) in enumerate(batch):
        N = points.shape[0]
        
        # 填充数据
        points_batch[i, :N, :] = points
        
        # 兼容处理：如果 target 是 [N]，则填入 [N, 1]；如果是 [N, 2]，则正常填入
        if len(target.shape) == 1:
            targets_batch[i, :N, 0] = target
        else:
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