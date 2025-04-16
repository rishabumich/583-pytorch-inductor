#!/usr/bin/env python
# coding: utf-8

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


# Define a custom module that uses soft_margin_loss
class MyModule(torch.nn.Module):
    def forward(self, *inputs):
        logits = inputs[0]
        targets = inputs[1]
        return (F.soft_margin_loss(logits, targets),)  # Use Soft Margin Loss in the forward pass

# Function to compute the loss (involving soft margin loss)
def compute_loss(logits, targets):
    # Compute soft margin loss
    loss = F.soft_margin_loss(logits, targets)
    return loss

# Example input tensors for logits and targets
logits = torch.randn(4, requires_grad=True)
targets = torch.tensor([1, -1, 1, -1], dtype=torch.float32)  # Targets must be -1 or 1

# Define a custom backend to print the decomposed graph
def print_backend(gm: GraphModule, inputs):
    print("Decomposed Graph:")
    print(gm.code)  # This will print the decomposed graph code
    return gm.forward  # Return the forward function for execution

# Use aot_module_simplified to compile the module and apply decompositions
compiled = aot_module_simplified(
    MyModule(),
    (logits, targets),
    fw_compiler=print_backend,  # Custom backend to print the graph
    decompositions=core_aten_decompositions(),  # Apply standard decompositions (including soft_margin_loss)
)

# Compute the loss by applying the forward pass (which triggers soft_margin_loss)
compute_loss(logits, targets)

# Function to print the decomposition of a given operator and its inputs
def print_operator_decomposition(operator_fn, *inputs):
    # Define a custom module that applies the operator
    class OperatorModule(torch.nn.Module):
        def forward(self, *inputs):
            return operator_fn(*inputs)  # Apply the passed operator in the forward method
    
    # Use aot_module_simplified to compile the module and apply decompositions
    def print_backend(gm: GraphModule, inputs):
        print("Decomposed Graph:")
        # Print out the graph in a human-readable format similar to the example you gave
        for node in gm.graph.nodes:
            # Use 'aten' ops (lower-level ops) for decomposition and ensure the output matches your format
            print(f"    {node.name} = {node.op}({', '.join(map(str, node.args))})")
        return gm.forward  # Return the forward function for execution

    # Compile the operator using AOT and print its decomposition
    compiled = aot_module_simplified(
        OperatorModule(),
        inputs,
        fw_compiler=print_backend,  # Custom backend to print the graph
        decompositions=core_aten_decompositions(),  # Apply standard decompositions
    )

# Example operators to decompose
def example_operator_1(logits, targets):
    return F.soft_margin_loss(logits, targets)

def example_operator_2(x):
    return F.relu(x)

def example_operator_3(x):
    return F.batch_norm(x, running_mean=torch.zeros_like(x), running_var=torch.ones_like(x))

# Example usage: Swap the operator function and inputs easily
if __name__ == "__main__":
    # Example 1: Soft Margin Loss
    logits = torch.randn(4, requires_grad=True)
    targets = torch.tensor([1, -1, 1, -1], dtype=torch.float32)  # Targets must be -1 or 1
    print("Decomposing Soft Margin Loss:")
    print_operator_decomposition(example_operator_1, logits, targets)
    
    # Example 2: ReLU
    x = torch.randn(4, requires_grad=True)
    print("\nDecomposing ReLU:")
    print_operator_decomposition(example_operator_2, x)
    
    # Example 3: Batch Normalization
    x_bn = torch.randn(4, 4, requires_grad=True)
    print("\nDecomposing BatchNorm:")



