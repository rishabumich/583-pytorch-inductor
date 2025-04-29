import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgEig_OutModule(torch.nn.Module):
    def forward(self, x, eigenvalues, eigenvectors):
        return torch.ops.aten.linalg_eig.out(x, eigenvalues, eigenvectors)

mod = Torch_Ops_Aten_LinalgEig_OutModule()

x = torch.randn(3)
eigenvalues = torch.randn(3)
eigenvectors = torch.randn(3)

args = (x, eigenvalues, eigenvectors,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
