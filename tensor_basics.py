# Tensor can have different dimensions
import torch


# This is a 1D tensor with 3 elements
x=torch.empty(3)
print(x)

# This is a 2D tensor with 2x3 elements 2-rows 3-columns
y=torch.empty(2,3)
print(y)

# This is a 3D tensor
z=torch.ones(2,2,3,dtype=torch.int)
print(z)
print(z.dtype)


# Some basic operations performed on tensors
a=torch.rand(2,2)
b=torch.rand(2,2)
c=a+b
print(c)


# Slicing in a tensors
d=torch.rand(5,3)
print(d)
print(d[:,1])
print(d[2,:])


# Adding requires grad argument
e=torch.ones(5,3,requires_grad=True)
print(e)