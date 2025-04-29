import torch

def test_repeat_interleave():
    # 1. Basic 1D input, int repeat
    x = torch.tensor([1, 2, 3])
    out = torch.repeat_interleave(x, 2)
    expected = torch.tensor([1, 1, 2, 2, 3, 3])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 2. Basic 2D input, flatten (no dim), int repeat
    y = torch.tensor([[1, 2], [3, 4]])
    out = torch.repeat_interleave(y, 2)
    expected = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 3. 2D input, repeat along dim=1
    out = torch.repeat_interleave(y, 3, dim=1)
    expected = torch.tensor([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 4, 4, 4]])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 4. 2D input, tensor repeats, dim=0
    repeats = torch.tensor([1, 2])
    out = torch.repeat_interleave(y, repeats, dim=0)
    expected = torch.tensor([[1, 2],
                              [3, 4],
                              [3, 4]])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 5. Same as above, specifying output_size explicitly
    out = torch.repeat_interleave(y, repeats, dim=0, output_size=3)
    expected = torch.tensor([[1, 2],
                              [3, 4],
                              [3, 4]])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 6. Only repeats as input: input = None, repeat tensor given
    r = torch.tensor([1, 2, 3])
    out = torch.repeat_interleave(r)
    expected = torch.tensor([0, 1, 1, 2, 2, 2])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 7. 3D input, repeat along a dimension
    z = torch.arange(8).reshape(2, 2, 2)
    out = torch.repeat_interleave(z, 2, dim=2)
    expected = torch.tensor([[[0, 0, 1, 1],
                              [2, 2, 3, 3]],
                             [[4, 4, 5, 5],
                              [6, 6, 7, 7]]])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 8. Repeats as broadcastable tensor
    input = torch.tensor([[1, 2], [3, 4]])
    repeats = torch.tensor([1, 2])
    out = torch.repeat_interleave(input, repeats, dim=1)
    expected = torch.tensor([[1, 2, 2],
                              [3, 4, 4]])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 9. Zero repeat case
    x = torch.tensor([5, 6, 7])
    repeats = torch.tensor([0, 2, 1])
    out = torch.repeat_interleave(x, repeats)
    expected = torch.tensor([6, 6, 7])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    # 10. Empty input tensor
    empty = torch.tensor([])
    out = torch.repeat_interleave(empty, 3)
    expected = torch.tensor([])
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

    print("All tests passed!")

if __name__ == "__main__":
    test_repeat_interleave()
