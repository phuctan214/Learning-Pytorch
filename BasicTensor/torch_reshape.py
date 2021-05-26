import torch

x = torch.arange(9)
x_view_33= x.view(3,3)
print(x_view_33)

y = x_view_33.t()
print(y)

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim= 0))

z = x1.view(-1)
print(z.shape)


batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)

z= x.permute(0,2,1)
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1))

print(x.unsqueeze(0).unsqueeze(1).shape)

a = torch.rand((3,4))
print(a.unsqueeze(0))# unsqueeze(0) convert (a,b) --> (1,a,b)
print(a.unsqueeze(1))# unsqueeze(1) conver(a,b) --> ()

b = torch.rand((3,4))
print(a.squeeze(0).shape)# squeeze(0) convert (1,a,b) --> (a,b)
print(a.squeeze(1).shape)# squeeze(1) convert (a,b,1) --> (a,b)



