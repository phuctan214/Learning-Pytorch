import torch

batch_size = 10
features = 25
x = torch.rand((batch_size,features))

print(x[0].shape) # x[0,:]
print(x[:,0].shape)
print(x[2,0:10])# 0:10 -> [0,1,2,3,4,5,...,9]

#Fancy indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x_rand = torch.rand((3,5))
rows = torch.tensor([1,0])
columns = torch.tensor([4,0])
print(x_rand[rows,columns])

#More advanced indexing
x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[x.remainder(2) == 0])

# Useful operation
print(torch.where(x > 5, x, x*2))
print(torch.tensor([7,6,3,23,6,7,9]).unique())
print(x.ndimension()) # 5,5,5 dimension is three
