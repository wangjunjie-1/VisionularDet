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
from VisionularDET.nanodet.evaluator import build_evaluator
from VisionularDET.nanodet.data.dataset import build_dataset
from VisionularDET.nanodet.data.collate import naive_collate
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import cv2
import logging

# 设置全局logger
logger = logging.getLogger('NanoDet')
log_level= logging.DEBUG
logger.setLevel(log_level)

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s - %(filename)s:%(lineno)d - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# 超参数
global_step = 0
device = 'cuda:0'

def get_device():
    """
    获取设备，优先使用GPU，如果不可用则使用CPU。
    """
    if torch.cuda.is_available():
        return torch.device('cuda:1')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def stack_batch_img(
    img_tensors: Sequence[torch.Tensor], divisible: int = 0, pad_value: float = 0.0
) -> torch.Tensor:
    '''
    Args:
        img_tensors (Sequence[torch.Tensor]):
        divisible (int):
        pad_value (float): value to pad

    Returns:
        torch.Tensor.
    '''
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
    """
    预处理一批输入数据，包括将图像移动到正确的设备上，并将它们堆叠成一个张量。

    Args:
        batch: 包含图像和其他输入数据的批次。

    Returns:
        预处理后的批次。
    """
    batch_imgs = batch['img']
    if isinstance(batch_imgs, list):
        batch_imgs = [img.to(device) for img in batch_imgs]
        batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
        batch['img'] = batch_img_tensor
    return batch


def train(model, train_loader, optimizer):
    """
    训练模型一个epoch。

    Args:
        model: 要训练的模型。
        train_loader: 训练数据加载器。
        optimizer: 优化器。
    """
    global global_step
    model.train()
    for idx, meta_info in enumerate(train_loader):
        global_step += 1

        # show_img = meta_info['img'][0]
        # breakpoint()
        # cv2.imwrite('tmp.png',show_img.to('cpu').numpy().transpose((1,2,0)))
        # 将一个batch的数据拼成tensor张量：BCHW
        meta_info = _preprocess_batch_input(meta_info)
        optimizer.zero_grad()
        head_out, loss, loss_states = model.forward_train(meta_info)
        loss.backward()
        optimizer.step()
        logger.info(f"Train|Iter({idx}/{len(train_loader)})|loss:{loss:.4f}")


def test(model,  test_loader):
    """
    测试模型。

    Args:
        model: 要测试的模型。
        test_loader: 测试数据加载器。

    Returns:
        测试结果的集合。
    """
    model.eval()
    all_results = {}
    with torch.no_grad():
        for meta_info in tqdm(test_loader, desc='Testing'):
            meta_info = _preprocess_batch_input(meta_info)
            pred, loss, loss_states = model.forward_train(meta_info)
            dets = model.head.post_process(pred, meta_info)
            all_results.update(dets)

    return all_results


def train_pipeline(model, train_loader, test_loader, optimizer, evaluator,
                num_epochs, save_flag=-10):
    """
    完整的训练流程。

    Args:
        model: 要训练的模型。
        train_loader: 训练数据加载器。
        test_loader: 测试数据加载器。
        optimizer: 优化器。
        evaluator: 评估器。
        num_epochs: 训练的epoch数。
        save_flag: 用于保存模型的标志。
    """
    all_results = test(model, test_loader)
    # 全局训练步数
    global global_step
    save_dir = Path(cfg.save_dir)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00005)
    for current_epoch in range(1, num_epochs + 1):
        
        # 训练
        train(model, train_loader, optimizer)
        
        # 调整学习率
        if global_step <= cfg.schedule.warmup.steps:
            logger.info('warm up阶段')
            if cfg.schedule.warmup.name == 'constant':
                k = cfg.schedule.warmup.ratio
            elif cfg.schedule.warmup.name == 'linear':
                k = 1 - (
                    1 - global_step / cfg.schedule.warmup.steps
                ) * (1 - cfg.schedule.warmup.ratio)
            elif cfg.schedule.warmup.name == 'exp':
                k = cfg.schedule.warmup.ratio ** (
                    1 - global_step / cfg.schedule.warmup.steps
                )
            else:
                raise Exception('Unsupported warm up type!')
            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * k
        else:
            logger.info('余弦退火阶段')
            scheduler.step()
        logger.info(
            f"Epoch:{current_epoch}/{num_epochs}| global_step:{global_step}|当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 验证
        if current_epoch % cfg.schedule.val_intervals != 0:
            continue
        all_results = test(model, test_loader)
        eval_results = evaluator.evaluate(all_results, save_dir)
        metric = eval_results[cfg.evaluator.save_key]
        # 保存最佳模型
        if metric > save_flag:
            save_flag = metric

            best_save_path = save_dir/'model_best'
            logger.info(f'Saving model to {best_save_path}')
            best_save_path.mkdir(exist_ok=True)

            path = best_save_path/'nanodet_model_best.pth'
            state_dict = model.state_dict()
            torch.save({'state_dict': state_dict}, path)

            path = best_save_path/'nanodet_model_best.pt'
            torch.save(model, path)
            
            # todo  
            # export onnx model
            

            # debug模式时保存一些额外信息
            if log_level == logging.DEBUG:
                txt_path = best_save_path/'eval_results.txt'
                with open(txt_path, 'a') as f:
                    f.write('Epoch:{}\n'.format(current_epoch + 1))
                    for k, v in eval_results.items():
                        f.write('{}: {}\n'.format(k, v))
        else:
            logger.warning(
                'Warning! Save_key is not in eval results! Only save model last!'
            )
        logger.info(eval_results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def main(args):
    load_config(cfg, args.config)

    # 设置随机种子
    torch.manual_seed(1115)
    # 创建对应文件夹
    Path(cfg.save_dir).mkdir(exist_ok=True,parents=True)

    # 保存训练日志
    global logger 
    file_handler = logging.FileHandler(f'{cfg.save_dir}/train.log') 
    file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 准备数据
    logger.info('Setting up data...')
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'test')
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

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

    # 创建模型
    logger.info('build model...')
    model_cfg = cfg.model
    name = model_cfg.arch.pop('name')
    model = NanoDetPlus(**model_cfg.arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练流程
    logger.info('start traing...')
    train_pipeline(model, train_dataloader, val_dataloader, optimizer, evaluator,
                num_epochs=100, save_flag=-10)


if __name__ == '__main__':
    args = parse_args()
    main(args)
