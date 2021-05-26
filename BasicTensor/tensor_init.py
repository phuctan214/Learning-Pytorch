import torch
import numpy as np

x = torch.empty(size=(3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5)
x = torch.arange(start= 0, end= 5, step= 1)
x = torch.linspace(start= 0, end= 1, steps= 10)
x = torch.empty(size= (1,5)).normal_(mean= 0, std = 1)
x = torch.empty(size=(1,5)).uniform_(0, 1)
print(x)

# how to conver them to different type
my_tensor = torch.arange(4)
print(my_tensor.bool())
print(my_tensor.int())# important int32
print(my_tensor.double())# important float64
print(my_tensor.long())
print(my_tensor.float())#important float32

np_array = np.ones((5,2))
np_to_tensor = torch.from_numpy(np_array)
bring_np_back = np_to_tensor.numpy()
print(np_to_tensor)
print(bring_np_back)
