import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Set_SourceTensorModule(torch.nn.Module):
    def forward(self, x, source):
        return torch.ops.aten.set_.source_Tensor(x, source)

mod = Torch_Ops_Aten_Set_SourceTensorModule()

x = torch.randn(3)
source = torch.randn(3)

args = (x, source,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
