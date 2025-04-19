import torch
import re

output_path = "signatures.txt"

schema_pattern = re.compile(r"^(aten::[^(]+)\((.*)\)$")

def parse_args(arg_str):
    parts = []
    curr, depth = "", 0
    for ch in arg_str:
        if ch == ',' and depth == 0:
            parts.append(curr.strip())
            curr = ""
        else:
            if ch in "([": depth += 1
            elif ch in ")]": depth -= 1
            curr += ch
    if curr:
        parts.append(curr.strip())

    names = []
    types = []

    for arg in parts:
        if arg == '*':
            continue
        match = re.match(r"([\w\[\]?!]+(?:\([^\)]*\))?)\s+(\w+)", arg)
        if match:
            typ, name = match.groups()
            names.append(name)
            types.append(typ)
        else:
            names.append(arg)
            types.append("unknown")

    return names, types

with open(output_path, "w", encoding="utf-8") as f:
    for op_name in dir(torch.ops.aten):
        if op_name.startswith("_"):
            continue

        try:
            op = getattr(torch.ops.aten, op_name)
            doc = getattr(op.op, "__doc__", None)
            if doc:
                schemas = [line.strip() for line in doc.splitlines() if line.strip().startswith("aten::")]
                for schema in schemas:
                    schema = schema.split("->")[0].strip()
                    match = schema_pattern.match(schema)
                    if match:
                        func_name, args_str = match.groups()
                        arg_names, arg_types = parse_args(args_str)
                        joined_names = ",".join(arg_names)              # no spaces
                        joined_types = ",".join(arg_types)
                        prefix = "torch.ops.aten."
                        func_name_clean = func_name.replace("aten::", "")
                        f.write(f"{prefix}{func_name_clean}({joined_names}){joined_types}\n")

        except Exception:
            continue

print("✅ Done. Output saved to:", output_path)
