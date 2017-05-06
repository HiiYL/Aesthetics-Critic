

l = torch.nn.Linear(6,1).cuda()
l2 = torch.nn.Linear(6,1).cuda()
input = Variable(torch.rand(10,6).cuda())
out = l(input)
target = Variable(torch.rand(10,1)).cuda()
crt = torch.nn.L1Loss().cuda()
loss = crt(out, target)
loss.backward()
print input.grad