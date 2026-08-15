"""
PPML C++ vs Python 性能对比
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import pyppml
import time
import numpy as np

def benchmark_inference():
    """对比 C++ 和 Python 推理速度"""
    
    # 创建模型
    config = pyppml.PPMLConfig()
    config.n_extra_blocks = 4
    config.n_main_blocks = 8
    config.n_refine_blocks = 4
    
    model = pyppml.PPMLModel(config)
    model.eval()
    
    # 测试不同序列长度
    seq_lengths = [64, 128, 256, 512, 1024]
    batch_size = 1
    n_seq = 64
    n_templ = 4
    
    results = []
    
    for L in seq_lengths:
        print(f"\nBenchmarking seq_len={L}...")
        
        # 构造输入
        input_data = pyppml.ModelInput()
        input_data.msa_latent = pyppml.numpy_to_tensor(
            np.random.randn(batch_size, n_seq, L, 164).astype(np.float32)
        )
        input_data.seq_tokens = pyppml.numpy_to_tensor(
            np.random.randint(0, 80, (batch_size, L)).astype(np.float32)
        )
        input_data.t1d = pyppml.numpy_to_tensor(
            np.random.randn(batch_size, n_templ, L, 80).astype(np.float32)
        )
        input_data.coords = pyppml.numpy_to_tensor(
            np.random.randn(batch_size, L, 3, 3).astype(np.float32)
        )
        
        # CPU 推理
        model.to(pyppml.Device.CPU)
        
        # warmup
        for _ in range(3):
            _ = model.forward(input_data)
        
        start = time.time()
        for _ in range(10):
            output = model.forward(input_data)
        cpu_time = (time.time() - start) / 10
        
        # CUDA 推理
        model.to(pyppml.Device.CUDA)
        input_data.msa_latent = input_data.msa_latent.cuda()
        input_data.seq_tokens = input_data.seq_tokens.cuda()
        input_data.t1d = input_data.t1d.cuda()
        input_data.coords = input_data.coords.cuda()
        
        # warmup
        for _ in range(3):
            _ = model.forward(input_data)
        
        # 同步
        import torch
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            output = model.forward(input_data)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / 10
        
        results.append({
            'seq_len': L,
            'cpu_time': cpu_time,
            'cuda_time': cuda_time,
            'speedup': cpu_time / cuda_time
        })
        
        print(f"  CPU: {cpu_time*1000:.1f}ms")
        print(f"  CUDA: {cuda_time*1000:.1f}ms")
        print(f"  Speedup: {cpu_time/cuda_time:.1f}x")
    
    # 打印汇总
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"{'Seq Len':<10} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Speedup':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['seq_len']:<10} {r['cpu_time']*1000:<12.1f} {r['cuda_time']*1000:<12.1f} {r['speedup']:<10.1f}x")

def benchmark_memory():
    """测试内存占用"""
    import torch
    
    config = pyppml.PPMLConfig()
    model = pyppml.PPMLModel(config)
    
    L = 256
    input_data = pyppml.ModelInput()
    input_data.msa_latent = pyppml.numpy_to_tensor(
        np.random.randn(1, 64, L, 164).astype(np.float32)
    ).cuda()
    
    torch.cuda.reset_peak_memory_stats()
    
    model.to(pyppml.Device.CUDA)
    output = model.forward(input_data)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"\nPeak CUDA memory: {peak_memory:.2f} GB")

if __name__ == "__main__":
    benchmark_inference()
    benchmark_memory()
