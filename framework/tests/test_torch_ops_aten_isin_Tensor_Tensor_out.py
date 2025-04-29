import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Isin_TensorTensorOutModule(torch.nn.Module):
    def forward(self, elements, test_elements, assume_unique, invert, out):
        return torch.ops.aten.isin.Tensor_Tensor_out(elements, test_elements, assume_unique, invert, out=out)

mod = Torch_Ops_Aten_Isin_TensorTensorOutModule()

elements = torch.randn(3)
test_elements = torch.randn(3)
assume_unique = True
invert = True
out = torch.empty(3)

args = (elements, test_elements, assume_unique, invert, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
