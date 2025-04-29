import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AddcmulModule(torch.nn.Module):
    def forward(self, x, tensor1, tensor2, value):
        return torch.ops.aten.addcmul(x, tensor1, tensor2, value)

mod = Torch_Ops_Aten_AddcmulModule()

x = torch.randn(3)
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
value = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, tensor1, tensor2, value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
