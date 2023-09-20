import torch

x = torch.randn(3,requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
print(z) 

z = z.mean()
print(z)

z.backward() #dz/dx
print(x.grad)