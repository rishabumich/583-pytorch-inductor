from torch._functorch.aot_autograd import aot_module_simplified
from torch._decomp import core_aten_decompositions

from torch.fx import GraphModule
import torch.nn.functional as F
import torch

class MyModule(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

mod = MyModule()
#example_input = (torch.randn(4),)
example_input = (torch.tensor([0.2, -0.5, 0.8, -1.2], requires_grad=True),torch.tensor([1, -1, 1, -1], dtype=torch.float32),)

def print_backend(gm: GraphModule, inputs):
    print("Decomposed Graph:")
    print(gm.code)  # <- This will show decomposed ops like mul, erf, etc.
    return gm.forward

compiled = aot_module_simplified(
    mod,
    example_input,
    fw_compiler=print_backend,
    decompositions=core_aten_decompositions(),
)

compiled(*example_input)