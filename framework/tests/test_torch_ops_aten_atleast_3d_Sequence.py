import torch
from torch._decomp import decomposition_table

op = torch.ops.aten.atleast_3d.Sequence

tensors = torch.randn(3)

print("Scalar-returning op:")
if op in decomposition_table:
    print(f"{op} is decomposed.")
else:
    print(f"{op} is NOT decomposed.")

result = op(tensors)
print("Result:", result)
