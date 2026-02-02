import calculator_api as my_math_api
import timeit
import random
import numpy as np


# Using the functions
result_mul = my_math_api.multiply(10.5, 2.0)
result_div = my_math_api.divide(10.0, 3.0)
new_val = my_math_api.add(5, 1)



def matrix_multiply(mat1, mat2):
    row1 = len(mat1)
    col1 = len(mat1[0])
    row2 = len(mat2)
    col2 = len(mat2[0])

    if col1 != row2:
        raise ValueError("Incompatible matrix dimensions for multiplication.")
    
    result = [[0.0 for _ in range(col2)] for _ in range(row1)]
    for i in range(row1):
        for j in range(col2):
            for k in range(col1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result




print(f"Multiplication: {result_mul}")
print(f"Division: {result_div}")
print(f"Added: {new_val}")
# print(f"Matrix multiplication result: {result}")


size = 128
A = [[random.random() for _ in range(size)] for _ in range(size)]
B = [[random.random() for _ in range(size)] for _ in range(size)]

iterations = 100

py_time = timeit.timeit(
    lambda: matrix_multiply(A, B), 
    number=iterations
)

# Time the C++ version
cpp_time = timeit.timeit(
    lambda: my_math_api.matrix_multiply(A, B), 
    number=iterations
)

C, D = np.array(A), np.array(B)

numpy_time = timeit.timeit(
    lambda: np.matmul(C, D), 
    number=iterations
)

strassen_time = timeit.timeit(
    lambda: my_math_api.strassen_matrix_multiply(A, B), 
    number=iterations
)

mat2d_A = my_math_api.matrix2d([elem for row in A for elem in row], size, size)
mat2d_B = my_math_api.matrix2d([elem for row in B for elem in row], size, size)

flatten_time = timeit.timeit(
    lambda: my_math_api.matrix_multiply_flatten(mat2d_A, mat2d_B), 
    number=iterations
)
print(f"\nFlattened C++ matrix multiplication time: {flatten_time:.6f} seconds")
print(f"Python matrix multiplication time: {py_time:.6f} seconds")
print(f"C++ matrix multiplication time: {cpp_time:.6f} seconds")
print(f"NumPy matrix multiplication time: {numpy_time:.6f} seconds")


print("\nPerformance Comparison:")

if py_time < cpp_time:
    print(f"Python implementation is faster by {cpp_time / py_time:.2f}x")

else:
    print(f"C++ implementation is faster by {py_time / cpp_time:.2f}x")


if numpy_time < cpp_time:
    print(f"NumPy implementation is faster then cpp by {cpp_time / numpy_time:.2f}x")
else:
    print(f"C++ implementation is faster by {numpy_time / cpp_time:.2f}x")

if numpy_time < py_time:
    print(f"NumPy implementation is faster then python by {py_time / numpy_time:.2f}x")
else:
    print(f"Python implementation is faster by {numpy_time / py_time:.2f}x")


print("\nStrassen's Matrix Multiplication:")
print(f"Strassen's C++ matrix multiplication time: {strassen_time:.6f} seconds")

if strassen_time < cpp_time:
    print(f"Strassen's C++ implementation is faster then standard C++ by {cpp_time / strassen_time:.2f}x")
else:
    print(f"Standard C++ implementation is faster by {strassen_time / cpp_time:.2f}x")

if strassen_time < numpy_time:
    print(f"Strassen's C++ implementation is faster then NumPy by {numpy_time / strassen_time:.2f}x")
else:
    print(f"NumPy implementation is faster by {strassen_time / numpy_time:.2f}x")


if flatten_time < cpp_time:
    print(f"Flattened C++ implementation is faster then standard C++ by {cpp_time / flatten_time:.2f}x")
else:
    print(f"Standard C++ implementation is faster by {flatten_time / cpp_time:.2f}x")

if flatten_time < numpy_time:
    print(f"Flattened C++ implementation is faster then NumPy by {numpy_time / flatten_time:.2f}x")
else:
    print(f"NumPy implementation is faster by {flatten_time / numpy_time:.2f}x")
