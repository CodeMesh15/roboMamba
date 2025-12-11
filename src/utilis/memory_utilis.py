
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MemoryEfficientWrapper(nn.Module):
    """Wrapper to add gradient checkpointing to any module"""
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs)


def optimize_model_memory(model, config):
    """Apply memory optimizations to model"""
    
    # 1. Gradient checkpointing
    if config.get('gradient_checkpointing', False):
        for name, module in model.named_modules():
            if 'MambaBlock' in str(type(module)):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, attr_name, MemoryEfficientWrapper(module))
    
    # 2. Enable activation checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # 3. Flash attention (if available)
    try:
        from flash_attn import flash_attn_func
        model.use_flash_attention = True
    except ImportError:
        pass
    
    return model


def get_optimal_batch_size(model, device, max_seq_len=512):
    """Automatically determine optimal batch size"""
    import torch.cuda as cuda
    
    if not cuda.is_available():
        return 8  # Default for CPU
    
    # Get available memory
    free_memory = cuda.mem_get_info()[0] / 1e9  # GB
    
    # Estimate memory per sample
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    dummy_text = torch.randint(0, 1000, (1, max_seq_len)).to(device)
    
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_input, dummy_text)
    
    memory_per_sample = cuda.max_memory_allocated() / 1e9
    
    # Calculate batch size (leave 20% buffer)
    optimal_batch_size = int((free_memory * 0.8) / memory_per_sample)
    
    return max(1, min(optimal_batch_size, 32))


def print_memory_stats():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
