import torch
import unittest
import numpy as np
from einops import repeat

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super(Rotary, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = repeat(freqs, "b n -> b (n j)", j=2).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

# rotary pos emb helpers:
def rotate_every_two(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).view(x.shape)

@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_every_two(x) * sin)

# define a unittest for the positional embedding layer
class ROPEUnitTest(unittest.TestCase):
    def test_shapes(self):
        # apply positional embedding
        test_input = torch.randn(16, 4, 64, 64)
        rope = Rotary(test_input.shape[-1])
        cos, sin = rope(test_input, seq_dim=2)
        output = apply_rotary_pos_emb(test_input, cos, sin)

        self.assertTrue(output.shape == test_input.shape, "Input shape and output shape must match")

    def test_rotation(self):
        # define a helper function
        def rotate_matrix(x, m:int):
            """Rotate a vector by theta 1 counter clockwise (theta 1 is the angle of the first 2 embeddings)"""
            assert x.shape == torch.Size([1, 2]), "'x' is not a 2-length vector"

            # compute components
            theta = 10000**(-2*(1 - 1)/2)
            sin_comp = np.sin(m * theta)
            cos_comp = np.cos(m * theta)
            rotation_matrix = torch.tensor([[cos_comp, sin_comp], [-sin_comp, cos_comp]]).float()

            return torch.matmul(x, rotation_matrix)
        
        # define the test case 1: [1, 0], [1, 0]
        test_input1 = torch.tensor([[1, 0], [0, 1]]).float()

        # compute using the positional embedding
        rope1 = Rotary(test_input1.shape[-1])
        cos1, sin1 = rope1(test_input1.unsqueeze(0).unsqueeze(1), seq_dim=2)
        module_output1 = apply_rotary_pos_emb(test_input1, cos1, sin1)

        # compute using rotational matrix
        expected_output1 = torch.cat(
            [rotate_matrix(test_input1[i].reshape(1, 2), i) for i in range(test_input1.shape[0])]
        )

        # test case 1
        self.assertTrue(torch.allclose(module_output1, expected_output1)), "Test 1 Failed: Vector is not rotated by theta1"

        # define the test case 2: random input of length 10
        test_input2 = torch.randn(10, 2)

        # compute using positional embedding
        rope2 = Rotary(test_input2.shape[-1])
        cos2, sin2 = rope2(test_input2.unsqueeze(0).unsqueeze(1), seq_dim=2)
        module_output2 = apply_rotary_pos_emb(test_input2, cos2, sin2)

        # compute using rotational matrix
        expected_output2 = torch.cat(
            [rotate_matrix(test_input2[i].reshape(1, 2), i) for i in range(test_input2.shape[0])]
        )

        # test case 2
        self.assertTrue(torch.allclose(module_output2, expected_output2)), "Test 2 Failed: Vector is not rotated by theta1"
        
if __name__ == "__main__":
    unittest.main()
