import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ReplicationPad2DModule(torch.nn.Module):
    def forward(self, x, padding):
        return torch.ops.aten.replication_pad2d(x, padding)

mod = Torch_Ops_Aten_ReplicationPad2DModule()

x = torch.randn(3)
padding = torch.tensor(0)  # Fallback for unknown type SymInt[4]

args = (x, padding,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
