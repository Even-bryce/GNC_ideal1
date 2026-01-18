import torch
from data_loader import build_dataloader
from my_model import get_model, get_loss, weighted_bce_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for points, targets in loader:
        """
        points : [B, N, 6]
        targets: [B, N, 1]
        """
        points = points.to(device)
        targets = targets.to(device)

        # [B, 6, N]
        points = points.permute(0, 2, 1)

        # xyz: [B, N, 3] —— 已归一化
        xyz = points[:, :3, :].permute(0, 2, 1)

        optimizer.zero_grad()

        logits = model(points)  # [B, N, 1]

        # 🔥 关键修改：直接把 xyz 传给 loss
        loss = criterion(logits, targets, xyz)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    """
    验证阶段：只看 BCE（不加几何正则）
    """
    model.eval()
    total_loss = 0.0

    for points, targets in loader:
        points = points.to(device)
        targets = targets.to(device)

        points = points.permute(0, 2, 1)
        logits = model(points)

        loss = weighted_bce_loss(logits, targets)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=1, input_dim=6).to(device)

    criterion = get_loss(
        w_bce=1.0,
        w_straight=0.1,
        w_safety=0.1,
        w_conn=0.0,
        # 你也可以在这里统一调几何超参
        delta_s=0.5,
        r_corridor=0.03,
        r_local=0.05,
        rho=1000.0
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    train_loader = build_dataloader(split='train')
    val_loader   = build_dataloader(split='val')

    for epoch in range(1, 101):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate(model, val_loader, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train: {train_loss:.4f} | Val(BCE): {val_loss:.4f}"
        )

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"ckpt_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
