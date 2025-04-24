import torch
from torch._decomp import decomposition_table

op = torch.ops.aten.acos.int

a = 3

print("Before decomposition:")
if op in decomposition_table:
    print(f"{op} is decomposed.")
else:
    print(f"{op} is NOT decomposed.")

print("After decomposition:")
print("Not applicable for scalar-returning ops.")

result = op(a)
print("Result:", result)
