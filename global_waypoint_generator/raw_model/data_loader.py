import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PathPointDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 加载数据
        data = np.load(self.data_files[idx])

        # 修正：对应 save_sample 的键名 'points' 和 'labels'
        # points shape: [N, 6] (前3维是xyz，后3维是特征)
        points = torch.from_numpy(data['points']).float() 
        
        # labels shape: [N, 1]
        target = torch.from_numpy(data['labels']).float()

        return points, target

def collate_fn(batch):
    """
    处理变长点云的 Padding
    返回:
        points_batch:  [B, N_max, 6]
        targets_batch: [B, N_max, 1]
    """
    # 获取当前 batch 中最大的点数 N_max
    # batch[i][0] 是 points, batch[i][1] 是 target
    N_max = max(item[0].shape[0] for item in batch)

    points_list = []
    targets_list = []

    for points, target in batch:
        N = points.shape[0]
        pad_n = N_max - N

        if pad_n > 0:
            # Padding Points: [N, 6] -> [N_max, 6] (补0)
            # 注意：最后3维特征补0不影响，前3维补0意味着原点，最好padding的点在mask里被忽略
            # 但 PointNet++ 对零填充通常具有鲁棒性
            points_pad = torch.cat([points, torch.zeros(pad_n, 6)], dim=0)
            
            # Padding Targets: [N, 1] -> [N_max, 1]
            target_pad = torch.cat([target, torch.zeros(pad_n, 1)], dim=0)
        else:
            points_pad = points
            target_pad = target

        points_list.append(points_pad)
        targets_list.append(target_pad)

    # 堆叠
    points_batch = torch.stack(points_list)    # [B, N_max, 6]
    targets_batch = torch.stack(targets_list)  # [B, N_max, 1]

    return points_batch, targets_batch

def build_dataloader(data_files, batch_size=8, shuffle=True, num_workers=0):
    # Windows下 num_workers 建议设为 0，Linux 可设为 4
    dataset = PathPointDataset(data_files)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )