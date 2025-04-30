import torch
from torch._decomp import decomposition_table

op = torch.ops.aten.sinh.float

a = None  # Fallback for unknown type |float

print("Scalar-returning op:")
if op in decomposition_table:
    print(f"{op} is decomposed.")
else:
    print(f"{op} is NOT decomposed.")

result = op(a)
print("Result:", result)
