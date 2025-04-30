import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ReplicationPad2D_OutModule(torch.nn.Module):
    def forward(self, x, padding, out):
        return torch.ops.aten.replication_pad2d.out(x, padding, out=out)

mod = Torch_Ops_Aten_ReplicationPad2D_OutModule()

x = torch.randn(3)
padding = torch.sym_int(3)
out = torch.empty(3)

args = (x, padding, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
