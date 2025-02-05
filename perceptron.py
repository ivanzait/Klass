import numpy as np

def activation(x):
	y = np.zeros(np.size(x))
	for i in range(len(x)): 
		if x[i]<0:
			y[i]=-1
		else:
			y[i]=1
	return y

def target(len,sigma,r):
	target_set={}
	for i in range(len):	
		x = np.random.uniform(-sigma,sigma)
		y = np.random.uniform(-sigma,sigma)
		if x**2 +y**2 < r:
			label = 1
		else:
			label = 0
		target_set[i] = [x,y,label]
	return  target_set
	
def guess(len,sigma):
	guess = {}
	for i in range(len): 
		x = np.random.uniform(-sigma,sigma)
		y = np.random.uniform(-sigma,sigma)
		guess[i] = [x,y]
	return guess

def init_weights():
        return np.random.normal(0,1,2)

n = 10 # number of points for training set

g = guess(4,1)
t = target(3,1,0.5)
#print("guess",g), print("target",t)

w1 = init_weights()
print("w1",w1),
z1 = np.dot(g[0],w1)
print('z1:',z1)
#a1=activation(z1)
#print('a1:',a1)


