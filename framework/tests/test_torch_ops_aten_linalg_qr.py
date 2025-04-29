import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgQrModule(torch.nn.Module):
    def forward(self, A, mode):
        return torch.ops.aten.linalg_qr(A, mode)

mod = Torch_Ops_Aten_LinalgQrModule()

A = torch.randn(3)
mode = torch.tensor(0)  # Fallback for unknown type str

args = (A, mode,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
