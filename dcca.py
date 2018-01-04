from __future__ import division
import torch
import numpy as np
from torch.autograd import Variable

max_iterations = 500
o,m = 10,20

View_1 = torch.nn.Sequential(
	torch.nn.Linear(20, 50),
	torch.nn.Sigmoid(),
	torch.nn.Linear(50, 50),
	torch.nn.Sigmoid(),
	torch.nn.Linear(50, 20)
)

View_2 = torch.nn.Sequential(
        torch.nn.Linear(20, 50),
        torch.nn.Sigmoid(),
        torch.nn.Linear(50, 50),
        torch.nn.Sigmoid(),
        torch.nn.Linear(50, 20)
)

learning_rate = 1e-4
optimizer1 = torch.optim.Adam(View_1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(View_2.parameters(), lr=learning_rate)

X1 = Variable(torch.FloatTensor(torch.rand(o,m)), requires_grad=True)
X2 = Variable(torch.FloatTensor(torch.rand(o,m)), requires_grad=True)

for iteration in xrange(max_iterations):
	Pred1 = View_1(X1)
	Pred2 = View_2(X2)
	
	#Pred1 = V1(X1)
	#Pred2 = V2(X2)
	
	H1 = torch.Tensor(Pred1.data)
	H2 = torch.Tensor(Pred2.data)

	ones = torch.ones(m,m)

	H1_hat = H1 - torch.mul(H1.mm(ones),(1.0/m))
	H2_hat = H2 - torch.mul(H2.mm(ones),(1.0/m))

	SigmaHat12 = torch.mul(H1_hat.mm(torch.t(H2_hat)), (1.0/m-1))
	SigmaHat11 = torch.mul(H1_hat.mm(torch.t(H1_hat)), (1.0/m-1))
	SigmaHat22 = torch.mul(H2_hat.mm(torch.t(H2_hat)), (1.0/m-1))

	U1, S1, V1 = torch.svd(SigmaHat11)
	U2, S2, V2 = torch.svd(SigmaHat22)

	SigmaHat11RootInv = torch.inverse(U1.mm(torch.pow(torch.diag(S1), 0.5)).mm(torch.t(V1)))
	SigmaHat22RootInv = torch.inverse(U2.mm(torch.pow(torch.diag(S2), 0.5)).mm(torch.t(V2)))

	T = SigmaHat11RootInv.mm(SigmaHat12).mm(SigmaHat22RootInv)

	corr = torch.trace(torch.t(T).mm(T)) ** 0.5
	if iteration % 10 == 0:
		print corr

	# Calculate d corr(H1,H2) / dH1
	A, D, G = torch.svd(T)
	#print A
	#print D
	#print G
	Delta11 = -SigmaHat11RootInv.mm(A).mm(torch.diag(D)).mm(torch.t(A)).mm(SigmaHat11RootInv)
	Delta12 =  SigmaHat11RootInv.mm(A).mm(torch.t(G)).mm(SigmaHat22RootInv)
	Delta22 = -SigmaHat22RootInv.mm(A).mm(torch.diag(D)).mm(torch.t(A)).mm(SigmaHat22RootInv)

	dH1 = torch.mul(torch.mul(Delta11.mm(H1_hat),2) + Delta12.mm(H2_hat), 1.0/(m -1))
	dH2 = torch.mul(torch.mul(Delta22.mm(H2_hat),2) + Delta12.mm(H1_hat), 1.0/(m -1))

	#print dH1
	#print dH2

	optimizer1.zero_grad()
	optimizer2.zero_grad()
	
	Pred1.backward(dH1)
	Pred2.backward(dH2)

	optimizer1.step()
	optimizer2.step()
