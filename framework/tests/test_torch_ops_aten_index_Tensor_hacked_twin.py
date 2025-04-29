import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Index_TensorHackedTwinModule(torch.nn.Module):
    def forward(self, x, indices):
        return torch.ops.aten.index.Tensor_hacked_twin(x, indices)

mod = Torch_Ops_Aten_Index_TensorHackedTwinModule()

x = torch.randn(3)
indices = torch.randn(3)

args = (x, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
