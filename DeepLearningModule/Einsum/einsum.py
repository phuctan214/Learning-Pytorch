import torch

x = torch.rand((3,2, 3))

# # Permutation of tensor
# torch.einsum("ij -> ji", x)
#
# # Summation
# torch.einsum("ij ->", x)
#
# # Column Sum
# torch.einsum("ij ->i", x)
#
# # Row Sum
# torch.einsum("ij ->j", x)
#
# # Matrix-Vector Multiplication
# v = torch.rand((1, 3))
# torch.einsum("ij,kj->ik", x, v)
#
# # Matrix-Matrix Multiplication
# torch.einsum("ij,kj->ik", x, x)  # 2*2
#
# # Dot product with first row of matrix
# torch.einsum("i,i->", x[0], x[0])
#
# # Dot product with matrix
# torch.einsum("ij,ij->", x, x)
#
# # Element-wise product
# torch.einsum("ij,ij->ij", x, x)
#
# # Outer product
# a = torch.rand((3))
# b = torch.rand((5))
# torch.einsum("i,j->ij", a, b)
#
# # Bacth Matrix Multiplition
# a = torch.rand((3,4,5))
# b = torch.rand((3,5,9))
# torch.einsum("ijk,ikl->ijl",a,b)
#
# #Matrix diagonal
# m = torch.rand((3,3))
# torch.einsum("ii->i",m)
#
# #Matrix trace
# torch.einsum("ii->",m)

a = torch.einsum("ijk->i(j*k)",x)
print(a)


