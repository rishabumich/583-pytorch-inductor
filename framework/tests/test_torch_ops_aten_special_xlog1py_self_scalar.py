import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SpecialXlog1Py_SelfScalarModule(torch.nn.Module):
    def forward(self, x, other):
        return torch.ops.aten.special_xlog1py.self_scalar(x, other)

mod = Torch_Ops_Aten_SpecialXlog1Py_SelfScalarModule()

x = None  # Fallback for unknown type |Scalar
other = torch.randn(3)

args = (x, other,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
