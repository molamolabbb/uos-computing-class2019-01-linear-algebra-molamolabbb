#!/usr/bin/env python3

from linalg import Vector, vector_sum, Matrix, identity
from math import sqrt
from pytest import approx

## Vector Tests

def test_vector_basic():
    v = Vector([1, 2, 3])
    assert v.i(0) == 1, "Hint: Vectors should be 0-indexed"
    assert v.i(1) == 2, "Hint: Vectors should be 0-indexed"
    assert v.i(2) == 3, "Hint: Vectors should be 0-indexed"
    assert v.shape() == 3

def to_list(v: Vector):
    return list(v.i(i) for i in range(v.shape()))
    
def test_vector_add():
    v1 = Vector([1, 2, 3])
    v2 = Vector([3, 4, 5])
    v12 = Vector.add(v1, v2)
    assert(to_list(v12) == [4, 6, 8]), "vector_addition failure"
    v1 = Vector([1, 2, 3, 4])
    v2 = Vector([3, 4, 5, 6])
    v12 = Vector.add(v1, v2)
    assert(to_list(v12) == [4, 6, 8, 10])
    assert(to_list(v1) == [1, 2, 3, 4]), "vector addition shouldn't change the input vectors"
    assert(to_list(v2) == [3, 4, 5, 6]), "vector addition shouldn't change the input vectors"

def test_vector_subtract():
    v1 = Vector([3, 4, 5])
    v2 = Vector([1, 2, 3])
    v12 = Vector.subtract(v1, v2)
    assert(to_list(v12) == [2, 2, 2])
    v1 = Vector([3, 4, 5, 6])
    v2 = Vector([1, 2, 3, 3])
    v12 = Vector.subtract(v1, v2)
    assert(to_list(v12) == [2, 2, 2, 3])

def test_vector_sum():
    v1 = Vector([1, 2, 3])
    v2 = Vector([3, 4, 5])
    v12 = vector_sum([v1, v2])
    assert(to_list(v12) == [4, 6, 8])
    v1 = Vector([1, 2, 3, 4])
    v2 = Vector([3, 4, 5, 6])
    v12 = vector_sum([v1, v2])
    assert(to_list(v12) == [4, 6, 8, 10])

def test_scalar_multiply():
    v = Vector([1, 2, 3, 4])
    assert(to_list(v) == to_list(v.scalar_multiply(1)))
    assert([2, 4, 6, 8] == to_list(v.scalar_multiply(2)))

def test_dot():
    v1 = Vector([1, 2, 3])
    v2 = Vector([3, 4, 5])
    v12 = dot(v1, v2)
    assert(to_list(dot(v1, v2)) == [3, 8, 15])

def test_dot():
    v1 = Vector([1, 2, 3])
    v2 = Vector([3, 4, 5])
    v12 = Vector.dot(v1, v2)
    assert(Vector.dot(v1, v2) == (3 + 8 + 15))

# def test_sum_of_squares():
#     assert(Vector.sum_of_squares(Vector[1]) == 1)
#     assert(Vector.sum_of_squares(Vector[1, 1]) == 2)
#     assert(Vector.sum_of_squares(Vector[1, 1, 1]) == 3)

def test_norm():
    assert(Vector.norm(Vector([1])) == 1)
    assert(Vector.norm(Vector([1, 1])) == approx(sqrt(2)))
    assert(Vector.norm(Vector([1, 1, 1])) == approx(sqrt(3)))

def test_distance():
    assert(Vector.distance(Vector([1]), Vector([0])) == 1)
    assert(Vector.distance(Vector([1, 1]), Vector([0, 1])) == 1)
    assert(Vector.distance(Vector([1, 1, 1]), Vector([0, 1, 1])) == 1)
    assert(Vector.distance(Vector([1, 1, 1]), Vector([0, 0, 1])) == approx(sqrt(2)))
    assert(Vector.distance(Vector([1, 1, 1]), Vector([0, 0, 0])) == approx(sqrt(3)))

## Matrix Tests

def test_matrix_shape():
    # assert(Matrix([]).shape() == (0,0))
    assert(Matrix([[1]]).shape() == (1,1))
    assert(Matrix([[1, 2, 3]]).shape() == (1,3))
    assert(Matrix([[1], [2], [3]]).shape() == (3,1))
    assert(Matrix([[1], [2], [3]]).shape() == (3,1))
    assert(Matrix([[1, 2, 3], [2, 2, 3], [3, 2, 3]]).shape() == (3,3))

def test_matrix_ij():
    assert(Matrix([[1]]).ij(0, 0) == 1)
    assert(Matrix([[2]]).ij(0, 0) == 2)
    assert(Matrix([[2, 3], [4, 5]]).ij(0, 0) == 2)
    assert(Matrix([[2, 3], [4, 5]]).ij(0, 1) == 3)
    assert(Matrix([[2, 3], [4, 5]]).ij(1, 0) == 4)
    assert(Matrix([[2, 3], [4, 5]]).ij(1, 1) == 5)

def test_matrix_identity():
    i1 = identity(1)
    assert i1.shape() == (1, 1)
    assert i1.ij(0, 0) == 1
    i2 = identity(2)
    assert i2.shape() == (2, 2)
    assert i2.ij(0, 0) == 1 and i2.ij(0, 1) == 0 and i2.ij(1, 1) == 1 and i2.ij(1, 0) == 0
    i3 = identity(3)
    assert i3.shape() == (3, 3)
    assert i3.ij(0, 0) == 1 and i3.ij(0, 1) == 0 and i3.ij(1, 1) == 1 and i3.ij(1, 0) == 0 and i3.ij(2, 2) == 1 and i3.ij(2, 0) == 0
    

def test_matrix_scalar_multiply():
    assert(Matrix([[1]]).scalar_multiply(1.).ij(0, 0) == 1.)
    assert(Matrix([[1]]).scalar_multiply(3.).ij(0, 0) == 3.)
    assert(Matrix([[1, 2]]).scalar_multiply(3.).ij(0, 1) == 6.)
    assert(Matrix([[1, 2], [4, 5]]).scalar_multiply(3.).ij(1, 1) == 15.)
    A = identity(2)
    B = A.scalar_multiply(3)
    assert A.shape() == (2, 2)
    assert A.ij(0, 0) == 1 and A.ij(0, 1) == 0 and A.ij(1, 1) == 1 and A.ij(0, 1) == 0, "scalar_multiply shouldn't change the self Matrix input"
    assert B.shape() == (2, 2)
    assert B.ij(0, 0) == 3 and B.ij(0, 1) == 0 and B.ij(1, 1) == 3 and B.ij(0, 1) == 0

def test_matrix_vector_multiply():
    assert(Matrix([[1]]).vector_multiply(Vector([1.])).i(0) == 1.)
    assert(Matrix([[1]]).vector_multiply(Vector([3.])).i(0) == 3.)
    assert(Matrix([[1]]).vector_multiply(Vector([3.])).shape() == 1)
    assert(Matrix([[1, 2]]).vector_multiply(Vector([1., 1.])).i(0) == 3.)
    assert(Matrix([[1, 2], [4, 5]]).vector_multiply(Vector([1., 1.])).i(1) == 9.)
    assert(Matrix([[1, 2], [4, 5]]).vector_multiply(Vector([1., 2.])).i(1) == 14.)
    assert(Matrix([[1, 2], [4, 5]]).vector_multiply(Vector([1., 2.])).shape() == 2)

def test_matrix_multiply():
    # 1x1 * 1x1 -> 1x1
    assert(Matrix([[1]]).multiply(Matrix([[1.]])).ij(0, 0) == 1.)
    assert(Matrix([[1]]).multiply(Matrix([[3.]])).ij(0, 0) == 3.)
    assert(Matrix([[1]]).multiply(Matrix([[3.]])).shape() == (1, 1))
    # 2x2 * 2x1 -> 2x1
    assert(Matrix([[1, 2], [4, 5]]).multiply(Matrix([[1.], [1.]])).ij(1, 0) == 9.)
    assert(Matrix([[1, 2], [4, 5]]).multiply(Matrix([[1.], [2.]])).ij(1, 0) == 14.)
    assert(Matrix([[1, 2], [4, 5]]).multiply(Matrix([[1.], [2.]])).shape() == (2, 1))
    # 2x2 * 2x2 -> 2x2
    A = identity(2)
    B = identity(2)
    C = Matrix.multiply(A, B)
    assert C.shape() == (2, 2)
    assert C.ij(0, 0) == 1 and C.ij(0, 1) == 0 and C.ij(1, 1) == 1 and C.ij(0, 1) == 0
    # 2x2 * 2x2 -> 2x2
    A = identity(2)
    B = identity(2).scalar_multiply(3)
    C = Matrix.multiply(A, B)
    assert A.shape() == (2, 2)
    assert A.ij(0, 0) == 1 and A.ij(0, 1) == 0 and A.ij(1, 1) == 1 and A.ij(0, 1) == 0
    assert B.shape() == (2, 2)
    assert B.ij(0, 0) == 3 and B.ij(0, 1) == 0 and B.ij(1, 1) == 3 and B.ij(0, 1) == 0
    assert C.shape() == (2, 2)
    assert C.ij(0, 0) == 3 and C.ij(0, 1) == 0 and C.ij(1, 1) == 3 and C.ij(0, 1) == 0
    # 3x3
    A = identity(3)
    i3 = A.multiply(A)
    assert i3.shape() == (3, 3)
    assert i3.ij(0, 0) == 1 and i3.ij(0, 1) == 0 and i3.ij(1, 1) == 1 and i3.ij(1, 0) == 0 and i3.ij(2, 2) == 1 and i3.ij(2, 0) == 0
