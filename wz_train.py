import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Sequence
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from VisionularDET.nanodet.arch import NanoDetPlus
from VisionularDET.nanodet.util import cfg, load_config
from VisionularDET.nanodet.data.dataset import build_dataset
from VisionularDET.nanodet.data.collate import naive_collate
# 2. 修改设备选择逻辑
def get_device():
    return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
device = get_device()


def stack_batch_img(
    img_tensors: Sequence[torch.Tensor], divisible: int = 0, pad_value: float = 0.0
) -> torch.Tensor:
    """
    Args:
        img_tensors (Sequence[torch.Tensor]):
        divisible (int):
        pad_value (float): value to pad

    Returns:
        torch.Tensor.
    """
    assert len(img_tensors) > 0
    assert isinstance(img_tensors, (tuple, list))
    assert divisible >= 0
    img_heights = []
    img_widths = []
    for img in img_tensors:
        assert img.shape[:-2] == img_tensors[0].shape[:-2]
        img_heights.append(img.shape[-2])
        img_widths.append(img.shape[-1])
    max_h, max_w = max(img_heights), max(img_widths)
    if divisible > 0:
        max_h = (max_h + divisible - 1) // divisible * divisible
        max_w = (max_w + divisible - 1) // divisible * divisible

    batch_imgs = []
    for img in img_tensors:
        padding_size = [0, max_w - img.shape[-1], 0, max_h - img.shape[-2]]
        batch_imgs.append(F.pad(img, padding_size, value=pad_value))
    return torch.stack(batch_imgs, dim=0).contiguous()


def _preprocess_batch_input(batch):
    batch_imgs = batch["img"]
    if isinstance(batch_imgs, list):
        batch_imgs = [img.to(device) for img in batch_imgs]
        batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
        batch["img"] = batch_img_tensor
    return batch

def train(model, device, train_loader, optimizer, epoch):
    model.train()   
    for batch_idx, meta_info in enumerate(train_loader):
        print(batch_idx)
        meta_info = _preprocess_batch_input(meta_info)
        optimizer.zero_grad()
        head_out, loss, loss_states = model.forward_train(meta_info)
        loss.backward()
        optimizer.step()

# 7. 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# 8. 添加学习率调度器和早停
def train_model(model, device, train_loader, test_loader, optimizer, 
                num_epochs, patience=3):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    best_acc = 0
    no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        train(model, device, train_loader, optimizer, epoch)  
        accuracy = test(model, device, test_loader)
        
        # 学习率调度
        scheduler.step(accuracy)
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
        
        # 早停
        if no_improve >= patience:
            print(f'Early stopping triggered after epoch {epoch}')
            break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    args = parser.parse_args()
    return args

# 主程序
def main(args):
    load_config(cfg, args.config)
    # 设置随机种子
    torch.manual_seed(42)
    global device
    print(f"使用设备: {device}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据
    print("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "test")
    # evaluator = build_evaluator(cfg.evaluator, val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )


    # 5. 初始化模型、损失函数和优化器
    model_cfg = cfg.model
    name = model_cfg.arch.pop("name")
    model =  NanoDetPlus(**model_cfg.arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 开始训练
    train_model(model, device, train_dataloader, val_dataloader, optimizer, 
                num_epochs=10, patience=3)


if __name__ == '__main__':
    args = parse_args()
    main(args)