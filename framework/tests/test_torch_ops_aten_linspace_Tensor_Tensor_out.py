import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Linspace_TensorTensorOutModule(torch.nn.Module):
    def forward(self, start, end, steps, out):
        return torch.ops.aten.linspace.Tensor_Tensor_out(start, end, steps, out=out)

mod = Torch_Ops_Aten_Linspace_TensorTensorOutModule()

start = torch.randn(3)
end = torch.randn(3)
steps = 3
out = torch.empty(3)

args = (start, end, steps, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
