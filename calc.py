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

strassen_matrix_view_time = timeit.timeit(
    lambda: my_math_api.strassen_entry(A, B), 
    number=iterations
)




print(f"\nFlattened C++ matrix multiplication time: {flatten_time:.6f} seconds")
print(f"Python matrix multiplication time: {py_time:.6f} seconds")
print(f"C++ matrix multiplication time: {cpp_time:.6f} seconds")
print(f"NumPy matrix multiplication time: {numpy_time:.6f} seconds")
print(f"Strassen's C++ matrix multiplication time: {strassen_time:.6f} seconds")
print(f"Strassen's C++ matrix multiplication with views time: {strassen_matrix_view_time:.6f} seconds")


def print_comparison(name1, time1, name2, time2):
    if time2 == 0: return # Avoid div by zero
    ratio = time1 / time2
    if ratio > 1:
        print(f"{name2} is {ratio:.2f}x faster than {name1}")
    else:
        print(f"{name1} is {1/ratio:.2f}x faster than {name2}")




print("\nPerformance Comparison:")

print_comparison("Python", py_time, "C++", cpp_time)
print_comparison("NumPy", numpy_time, "C++", cpp_time)
print_comparison("NumPy", numpy_time, "Python", py_time)
print_comparison("Strassen's C++", strassen_time, "C++", cpp_time)
print_comparison("Strassen's C++", strassen_time, "NumPy", numpy_time)
print_comparison("Flattened C++", flatten_time, "C++", cpp_time)
print_comparison("Flattened C++", flatten_time, "NumPy", numpy_time)
print_comparison("Strassen's C++ with views", strassen_matrix_view_time, "C++", cpp_time)
print_comparison("Strassen's C++ with views", strassen_matrix_view_time, "NumPy", numpy_time)
print_comparison("Strassen's C++ with views", strassen_matrix_view_time, "Strassen's C++", strassen_time)
print_comparison("Strassen's C++ with views", strassen_matrix_view_time, "Python", py_time)