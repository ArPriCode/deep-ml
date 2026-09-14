import torch
import torch.nn as nn

class LoopedStack(nn.Module):
    def __init__(self, blocks, n_loops):
        super().__init__()
        # Store blocks in an nn.ModuleList so PyTorch tracks their parameters
        if isinstance(blocks, nn.ModuleList):
            self.blocks = blocks
        else:
            self.blocks = nn.ModuleList(blocks)
            
        self.n_loops = n_loops

    def forward(self, x, n_loops=None):
        loops = n_loops if n_loops is not None else self.n_loops
        
        # Apply the blocks sequentially for the specified number of loop passes
        for _ in range(loops):
            for block in self.blocks:
                x = block(x)
                
        return x

    def block_applications(self, n_loops=None):
        loops = n_loops if n_loops is not None else self.n_loops
        return len(self.blocks) * loops

    def shared_parameter_count(self):
        # Total parameters stored across all blocks (each parameter counted once)
        return sum(p.numel() for p in self.parameters())

    def unrolled_parameter_count(self, n_loops=None):
        # Count of parameters if every block application used distinct parameters
        loops = n_loops if n_loops is not None else self.n_loops
        return self.shared_parameter_count() * loops
        