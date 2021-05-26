import torch

x = torch.tensor([1,2,3])
y = torch.tensor([7,8,9])

#Addition
z = x + y

#Subtraction
z = x - y

#Division
z = torch.true_divide(x,y)

#Inplace operation
t = torch.zeros((3,3))
t.add_(x)
t = t + x
print(t)

#Exponentiation
m = x**2
print(m)

#Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)
print(x3)

#Matrix Exponentiation
x1 = torch.ones((5,5))
print(x1.matrix_power(3))

#Element Wise
z = x*y
print(z)

#Dot Operation
z = torch.dot(x,y)
print(z)

#Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((bacth,m,p))
out_bmm = torch.bmm(tensor1,tensor2)

#Example of Broadcasting

#Other useful operation
sum_x = torch.sum(x,dim=0)
max_num, max_index = torch.max(x, dim=0)
min_num, min_index = torch.min(x,dim=0)
abs_x = torch.abs(x)
mean_x = torch.mean(x.float(), dim=0)
sorted_y, indices = torch.sort(y, dim= 0, descending= True)
z = torch.clamp(x, min= 0, max= 10)



