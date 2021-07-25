import numpy as np;
import torch
import io

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

my_values = {
    'a': torch.ones(2, 2),
    'b': torch.ones(2, 2) + 10,
    'c': 'hello',
    'd': 6
}

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
container = torch.jit.script(Container(my_values))
container.save("../build/fromPython.pt")

b = torch.jit.load("../build/fromCPP.pt")
x = list(b.parameters())
print(x)