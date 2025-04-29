import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BlockDiag_OutModule(torch.nn.Module):
    def forward(self, tensors, out):
        return torch.ops.aten.block_diag.out(tensors, out=out)

mod = Torch_Ops_Aten_BlockDiag_OutModule()

tensors = torch.randn(3)
out = torch.empty(3)

args = (tensors, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
