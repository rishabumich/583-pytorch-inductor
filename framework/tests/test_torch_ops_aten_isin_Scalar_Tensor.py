import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Isin_ScalarTensorModule(torch.nn.Module):
    def forward(self, element, test_elements, assume_unique, invert):
        return torch.ops.aten.isin.Scalar_Tensor(element, test_elements, assume_unique, invert)

mod = Torch_Ops_Aten_Isin_ScalarTensorModule()

element = torch.tensor(0)  # Fallback for unknown type |Scalar
test_elements = torch.randn(3)
assume_unique = True
invert = True

args = (element, test_elements, assume_unique, invert,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
