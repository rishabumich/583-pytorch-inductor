import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgQr_OutModule(torch.nn.Module):
    def forward(self, A, mode, Q, R):
        return torch.ops.aten.linalg_qr.out(A, mode, Q, R)

mod = Torch_Ops_Aten_LinalgQr_OutModule()

A = torch.randn(3)
mode = torch.tensor(0)  # Fallback for unknown type str
Q = torch.randn(3)
R = torch.randn(3)

args = (A, mode, Q, R,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
