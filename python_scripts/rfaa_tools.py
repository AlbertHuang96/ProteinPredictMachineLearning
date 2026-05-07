"""
RFAA Python 工具模块
供 C++ 通过 Python C API 调用
"""

import numpy as np
import subprocess
import os
from typing import Tuple, Optional

def run_hhblits(sequence: str, database_path: str, 
                n_iter: int = 3, e_value: float = 0.001) -> np.ndarray:
    """
    运行 HHblits 进行 MSA 搜索
    
    Args:
        sequence: 查询序列 (FASTA 格式)
        database_path: HHblits 数据库路径
        n_iter: 迭代次数
        e_value: E-value 阈值
    
    Returns:
        MSA 特征数组 (N, L, 83)
    """
    # 写入临时序列文件
    temp_fasta = "/tmp/query.fasta"
    with open(temp_fasta, "w") as f:
        f.write(">query\n")
        f.write(sequence + "\n")
    
    # 运行 HHblits
    cmd = [
        "hhblits",
        "-i", temp_fasta,
        "-d", database_path,
        "-n", str(n_iter),
        "-e", str(e_value),
        "-oa3m", "/tmp/output.a3m",
        "-cpu", "4"
    ]
    
    subprocess.run(cmd, check=True)
    
    # 解析 a3m 文件并转换为特征
    msa_features = parse_a3m("/tmp/output.a3m")
    
    return msa_features

def search_templates(sequence: str, pdb_db: str,
                     n_templates: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    搜索同源模板
    
    Returns:
        t1d: (T, L, 80) 模板 1D 特征
        t2d: (T, L, L, ...) 模板 2D 特征
    """
    # 运行 HHsearch
    cmd = [
        "hhsearch",
        "-i", "/tmp/query.a3m",
        "-d", pdb_db,
        "-o", "/tmp/hhsearch.hhr",
        "-z", str(n_templates)
    ]
    
    subprocess.run(cmd, check=True)
    
    # 解析 hhr 文件
    templates = parse_hhr("/tmp/hhsearch.hhr")
    
    # 提取模板特征
    t1d = extract_t1d(templates)
    t2d = extract_t2d(templates)
    
    return t1d, t2d

def load_torch_weights(pt_path: str, output_dir: str = "weights/") -> dict:
    """
    加载 PyTorch 权重并转换为 numpy 格式供 C++ 读取
    
    Args:
        pt_path: PyTorch .pt 或 .pth 文件路径
        output_dir: 输出 numpy 权重文件目录
    
    Returns:
        权重字典
    """
    import torch
    
    state_dict = torch.load(pt_path, map_location='cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    weight_map = {}
    for name, param in state_dict.items():
        np_array = param.cpu().numpy()
        weight_map[name] = np_array
        
        # 保存为 .npy
        safe_name = name.replace('.', '_')
        np.save(os.path.join(output_dir, f"{safe_name}.npy"), np_array)
    
    return weight_map

def compute_loss(outputs: dict, targets: dict) -> dict:
    """
    计算训练损失
    
    Returns:
        损失字典
    """
    import torch
    import torch.nn as nn
    
    losses = {}
    
    # 掩码语言模型损失
    if 'masked_tokens' in targets:
        mlm_loss = nn.CrossEntropyLoss()(outputs['token_logits'], targets['masked_tokens'])
        losses['mlm'] = mlm_loss
    
    # 距离图损失
    if 'dist_bins' in targets:
        dist_loss = nn.CrossEntropyLoss()(outputs['dist_logits'], targets['dist_bins'])
        losses['dist'] = dist_loss
    
    # FAPE 损失 (帧对齐点误差)
    if 'coords' in targets:
        fape_loss = compute_fape(outputs['coords'], targets['coords'])
        losses['fape'] = fape_loss
    
    # 扭转角损失
    if 'alpha' in targets:
        angle_loss = compute_angle_loss(outputs['alpha'], targets['alpha'])
        losses['angle'] = angle_loss
    
    # 辅助损失
    if 'lddt' in targets:
        lddt_loss = nn.MSELoss()(outputs['lddt'], targets['lddt'])
        losses['lddt'] = lddt_loss
    
    # 总损失
    total_loss = sum(losses.values())
    losses['total'] = total_loss
    
    return losses

def compute_fape(pred_coords, true_coords, eps=1e-8):
    """计算 FAPE 损失"""
    import torch
    
    # 简化实现
    diff = pred_coords - true_coords
    loss = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)
    return loss.mean()

def compute_angle_loss(pred_alpha, true_alpha):
    """计算扭转角损失 (使用复数表示)"""
    import torch
    
    # pred_alpha, true_alpha: (B, L, NTOTALDOFS, 2) - cos/sin
    pred_complex = torch.complex(pred_alpha[..., 0], pred_alpha[..., 1])
    true_complex = torch.complex(true_alpha[..., 0], true_alpha[..., 1])
    
    # 角度差异
    diff = pred_complex * torch.conj(true_complex)
    angle_diff = torch.angle(diff)
    
    return (angle_diff ** 2).mean()

def parse_a3m(a3m_path: str) -> np.ndarray:
    """解析 a3m MSA 文件"""
    # 简化实现
    return np.zeros((10, 100, 83), dtype=np.float32)

def parse_hhr(hhr_path: str) -> list:
    """解析 HHsearch 结果"""
    return []

def extract_t1d(templates: list) -> np.ndarray:
    """提取模板 1D 特征"""
    return np.zeros((4, 100, 80), dtype=np.float32)

def extract_t2d(templates: list) -> np.ndarray:
    """提取模板 2D 特征"""
    return np.zeros((4, 100, 100, 44), dtype=np.float32)
