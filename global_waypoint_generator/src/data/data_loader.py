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

        # points shape: [N, D] (前3维是xyz，后D-3维是特征)
        points = torch.from_numpy(data['points']).float() 
        
        # labels shape: [N, 1]
        target = torch.from_numpy(data['labels']).float()

        # ==========================================
        # [新增] 提取真实航路点 (包含起终点)
        # waypoints shape: [M, 3] (M 是不固定的)
        # ==========================================
        waypoints = torch.from_numpy(data['waypoints']).float()

        return points, target, waypoints

def collate_fn(batch):
    """
    处理变长点云的 Padding
    返回:
        points_batch:  [B, N_max, D]
        targets_batch: [B, N_max, 1]
        waypoints_list: 一个长度为 B 的 list，里面每个元素是 [M, 3] 的 Tensor
    """
    # batch[i][0] 是 points, batch[i][1] 是 target, batch[i][2] 是 waypoints
    N_max = max(item[0].shape[0] for item in batch)

    points_list = []
    targets_list = []
    waypoints_list = []  # [新增] 专门用来装变长航路点

    for points, target, waypoints in batch:
        N = points.shape[0]
        pad_n = N_max - N

        if pad_n > 0:
            C = points.shape[1] 
            points_pad = torch.cat([points, torch.zeros(pad_n, C, device=points.device)], dim=0)
            target_pad = torch.cat([target, torch.zeros(pad_n, 1)], dim=0)
        else:
            points_pad = points
            target_pad = target

        points_list.append(points_pad)
        targets_list.append(target_pad)
        
        # [新增] 直接把没做任何修改的 waypoints 塞进列表
        waypoints_list.append(waypoints)

    # 堆叠网络需要的输入
    points_batch = torch.stack(points_list)    # [B, N_max, D]
    targets_batch = torch.stack(targets_list)  # [B, N_max, 1]

    # 注意这里返回了三个变量！
    return points_batch, targets_batch, waypoints_list

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