"""
PPML 训练脚本 (Python 端)
与 C++ 框架配合进行混合训练
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import pyppml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class PPMLDataLoader:
    """数据加载器：从 C++ 获取批次数据"""
    
    def __init__(self, data_dir, batch_size=2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
    def __iter__(self):
        # 加载数据并转换为 C++ Tensor
        for batch_file in sorted(os.listdir(self.data_dir)):
            data = np.load(os.path.join(self.data_dir, batch_file))
            
            # 转换为 pyppml.Tensor
            input_data = pyppml.ModelInput()
            input_data.msa_latent = pyppml.numpy_to_tensor(data['msa_latent'])
            input_data.seq_tokens = pyppml.numpy_to_tensor(data['seq_tokens'])
            input_data.t1d = pyppml.numpy_to_tensor(data['t1d'])
            input_data.coords = pyppml.numpy_to_tensor(data['coords'])
            
            targets = {
                'coords': data['true_coords'],
                'alpha': data['true_alpha'],
                'dist_bins': data['dist_bins']
            }
            
            yield input_data, targets

def train_with_cpp_model():
    """使用 C++ 模型进行训练"""
    
    # 创建 C++ 模型
    config = pyppml.PPMLConfig()
    config.d_msa = 256
    config.d_pair = 128
    config.d_state = 32
    config.n_extra_blocks = 4
    config.n_main_blocks = 8
    config.n_refine_blocks = 4
    
    model = pyppml.PPMLModel(config)
    model.to(pyppml.Device.CUDA)
    model.train()
    
    # 加载预训练权重
    if os.path.exists("ppml_weights.bin"):
        model.load_weights("ppml_weights.bin")
    
    # 优化器 (Python 端)
    # 注意：这里需要自定义优化器或把梯度传回 Python
    optimizer = torch.optim.Adam([], lr=1e-4)  # 占位
    
    # 数据加载
    dataloader = PPMLDataLoader("data/train", batch_size=2)
    
    # 训练循环
    for epoch in range(10):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_data, targets) in enumerate(dataloader):
            # C++ 前向传播
            output = model.forward(input_data)
            
            # 将输出转回 PyTorch 计算损失
            pred_coords = torch.from_numpy(output.coords.to_numpy())
            pred_alpha = torch.from_numpy(output.alpha.to_numpy())
            
            true_coords = torch.from_numpy(targets['coords'])
            true_alpha = torch.from_numpy(targets['alpha'])
            
            # 计算损失
            loss = compute_loss(pred_coords, pred_alpha, true_coords, true_alpha)
            
            # 反向传播 (需要 C++ 支持梯度回传)
            # 或者：在 Python 端维护一份模型，用 distillation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")
        
        # 保存权重
        model.save_weights(f"ppml_weights_epoch_{epoch}.bin")
    
    # 导出 ONNX
    exporter = pyppml.ONNXExporter()
    onnx_config = pyppml.ONNXExportConfig()
    onnx_config.output_path = "ppml_model.onnx"
    onnx_config.opset_version = 17
    
    exporter.export_model(model, onnx_config)
    print("ONNX model exported.")

def compute_loss(pred_coords, pred_alpha, true_coords, true_alpha):
    """计算组合损失"""
    # FAPE 损失
    fape = torch.sqrt(((pred_coords - true_coords) ** 2).sum(-1) + 1e-8).mean()
    
    # 扭转角损失
    pred_complex = torch.complex(pred_alpha[..., 0], pred_alpha[..., 1])
    true_complex = torch.complex(true_alpha[..., 0], true_alpha[..., 1])
    angle_loss = (torch.angle(pred_complex * torch.conj(true_complex)) ** 2).mean()
    
    return fape + 0.5 * angle_loss

if __name__ == "__main__":
    train_with_cpp_model()
