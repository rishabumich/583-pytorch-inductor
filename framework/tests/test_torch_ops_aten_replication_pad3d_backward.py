import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ReplicationPad3DBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, padding):
        return torch.ops.aten.replication_pad3d_backward(grad_output, x, padding)

mod = Torch_Ops_Aten_ReplicationPad3DBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
padding = torch.tensor(0)  # Fallback for unknown type SymInt[6]

args = (grad_output, x, padding,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
