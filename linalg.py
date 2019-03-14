#!/usr/bin/env python3

from math import sqrt

# WRITE YOUR CODE FOR VECTOR AND MATRIX  HERE

if __name__ == "__main__":
    #
    # WRITE CODE TO DO CHECKS AS YOU GO HERE
    #
    # you can run pytest also, but it will failure if you haven't at
    # least setup the Vector, Matrix classes and the vector_sum and
    # identity function (can be blank, but must exist)
    #
    # Anything in here won't be run when importing
    #
    print("Running linalg on the command line")


class Vector:
	def __init__(self, initial=[0,0,0]):
		self.v = initial

	def shape(self):
		return len(self.v)
		
	def i(self, ith=0):
		return float(self.v[ith])

	def add(self, w):
		a = [] 
		for i in range(self.shape()):
			a.append(self.v[i]+w.i(i))
		return Vector(a)

	def subtract(self, w):
		s = []
		for i in range(self.shape()): 
			s.append(self.v[i]-w.i(i))
		return Vector(s)

	def scalar_multiply(self, c):
		scalar = []
		for i in range(self.shape()):
			scalar.append(c*self.v[i])
		return Vector(scalar)

	def mean(self):
		return sum([self.v[i] for i in range(len(self.v))])/len(self.v)

	def dot(self, w):
		d = 0
		for i in range(self.shape()): 
			d+=(self.v[i]*w.i(i))
		return d

	def norm(self):
		n = 0
		for i in range(self.shape()):
			n+=(self.v[i]**2)
		return sqrt(n)

	def distance(self, w):
		return self.subtract(w).norm()
		
	

	

if __name__ == "__main__":
	vec = Vector([1,2,3])
	print("x = ", vec.i(0))

	vec2 = Vector([3,4,5])
	


def vector_sum(l):
	vec_sum = l[0]
	for i in range(len(l)):
		if i != 0 :
		    vec_sum = vec_sum.add(l[i])
	return vec_sum


class Matrix:
	def __init__(self, m=[[1,2],[3,4]]):
		self.m = m

	def shape(self):
		return (len(self.m), len(self.m[0]))

	def ij(self, i, j):
		return self.m[i][j]

	def scalar_multiply(self, c):
		M1 = []
		for i in range(self.shape()[0]):
			M2 = []
			for j in range(self.shape()[1]):
				M2.append(c*self.m[i][j])
			M1.append(M2)
		return Matrix(M1)

	def vector_multiply(self, v):
		O =list(0 for i in range(self.shape()[0]))
		for i in range(self.shape()[0]):
			for j in range(v.shape()):
				O[i] += self.m[i][j]*v.i(j)
		return Vector(O)

	def transpose(self):
		O1 = []
		for i in range(self.shape()[0]):
			O2 = []
			for j in range(self.shape()[0]):
				O2.append(self.m[j][i])
			O1.append(O2)
		return O1

	def multiply(self, B):
		O1 =list((list(0 for j in range(B.shape()[1]))) for i in range(self.shape()[0]))
		for i in range(self.shape()[0]):
			for k in range(self.shape()[1]):
				for j in range(B.shape()[1]):
					O1[i][j] += (self.m[i][k]*B.m[k][j])
				
		return Matrix(O1)
			
def identity(n):
	I = []
	for i in range(n):
		for j in range(n):
			if i==j: I.append(1)
			else : I.append(0)
	return Matrix([I[i*n:(i+1)*n] for i in range(n)])

	
