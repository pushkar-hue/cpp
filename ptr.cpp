#include  <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;
typedef std::vector<std::vector<float>> Matrix;
typedef std::vector<float> Vector;

struct matrix2d
{
    std::vector<float> v_;
    size_t x_, y_;
	matrix2d(std::vector<float> v, size_t x, size_t y) : v_(v), x_(x), y_(y) {}
};


// Helper to perform valid pointer arithmetic
// A view is defined by:
// - A pointer to the top-left corner
// - The dimension of the sub-matrix (n)
// - The 'stride' (how far to jump in memory to get to the next row)
struct MatrixView {
    const float* data;
    size_t stride; 
    size_t n;

    // Helper to access elements: view[i][j]
    float get(size_t row, size_t col) const {
        return data[row * stride + col];
    }
};


// Adds two views and returns a NEW flat vector
std::vector<float> add_views(MatrixView A, MatrixView B) {
    std::vector<float> result(A.n * A.n);
    for (size_t i = 0; i < A.n; i++) {
        for (size_t j = 0; j < A.n; j++) {
            // Standard flat indexing for the result (stride = n)
            // Stride-based indexing for inputs
            result[i * A.n + j] = A.get(i, j) + B.get(i, j);
        }
    }
    return result;
}

// Subtracts two views
std::vector<float> sub_views(MatrixView A, MatrixView B) {
    std::vector<float> result(A.n * A.n);
    for (size_t i = 0; i < A.n; i++) {
        for (size_t j = 0; j < A.n; j++) {
            result[i * A.n + j] = A.get(i, j) - B.get(i, j);
        }
    }
    return result;
}

// Helper: Convert nested Matrix to flat std::vector
std::vector<float> flatten_to_vector(const Matrix& mat) {
    std::vector<float> flat;
    if (mat.empty()) return flat;
    
    size_t n = mat.size();
    flat.reserve(n * n); // Pre-allocate memory for speed
    
    for (const auto& row : mat) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

Matrix matrix_multiply_flatten(const matrix2d &m1, const matrix2d &m2)
{
    if (m1.x_ != m2.y_) {
        throw std::invalid_argument("Incompatible matrix dimensions.");
    }
    
    Matrix result(m1.y_, std::vector<float>(m2.x_, 0.0f));

	#pragma omp parallel for
    for (size_t i = 0; i < m1.y_; i++) {
        for (size_t k = 0; k < m1.x_; k++) {
            // Grab the element from m1 once for the entire J loop
            float val1 = m1.v_[i * m1.x_ + k];
            
            for (size_t j = 0; j < m2.x_; j++) {
                // We add to the result row incrementally
                // This accesses result[i][j] and m2.v_[...] linearly!
                result[i][j] += val1 * m2.v_[k * m2.x_ + j];
            }
        }
    }
    return result;
}



Matrix createMatrix(int n){
	return Matrix(n, std::vector<float>(n, 0.0f));	
}


Matrix addMatrices(const Matrix& mat1, const Matrix& mat2){
	int n = mat1.size();
	Matrix result = createMatrix(n);

	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j){
			result[i][j] = mat1[i][j] + mat2[i][j];
		}
	}
	return result;
}

Matrix subtractMatrices(const Matrix& mat1, const Matrix& mat2){
	int n = mat1.size();
	Matrix result = createMatrix(n);

	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j){
			result[i][j] = mat1[i][j] - mat2[i][j];
		}
	}
	return result;
}

std::vector<std::vector<float>> matrix_multiply(
	const std::vector<std::vector<float>>& mat1,
	const std::vector<std::vector<float>>& mat2)
{
	int row1 = mat1.size();
	int col1 = mat1[0].size();
	int row2 = mat2.size();
	int col2 = mat2[0].size();

	if (col1 != row2) {
		throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
	}

	std::vector<std::vector<float>> result(row1, std::vector<float>(col2, 0.0f));

	for (int i = 0; i < row1; ++i){
		for (int j = 0; j < col2; ++j){
			for (int k = 0; k < col1; ++k){
				result[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	return result;
}

// Forward declaration of your leaf solver
Matrix matrix_multiply_flatten(const matrix2d &m1, const matrix2d &m2);

// The core recursive function
Matrix strassen_recursive(MatrixView A, MatrixView B) {
    size_t n = A.n;

    // Base Case: Switch to standard loop-based multiply
    // Tuning: n <= 64 or 128 is usually optimal
    if (n <= 64) {
        // We must copy the View data into your matrix2d format for the leaf function
        // (This copy is small and fast compared to the large allocations)
        std::vector<float> a_vec(n * n);
        std::vector<float> b_vec(n * n);
        
        for(size_t i=0; i<n; i++) {
            for(size_t j=0; j<n; j++) {
                a_vec[i*n + j] = A.get(i, j);
                b_vec[i*n + j] = B.get(i, j);
            }
        }
        
        // Use your existing optimized leaf function
        return matrix_multiply_flatten(matrix2d(a_vec, n, n), matrix2d(b_vec, n, n));
    }

    size_t newSize = n / 2;

    // --- CREATE VIEWS (Zero Allocation) ---
    // We just calculate where the quadrants start in memory.
    
    // A11 starts at A.data
    MatrixView a11 = { A.data,                        A.stride, newSize };
    // A12 starts at A.data + newSize
    MatrixView a12 = { A.data + newSize,              A.stride, newSize };
    // A21 starts at A.data + (row_jump * newSize)
    MatrixView a21 = { A.data + (A.stride * newSize), A.stride, newSize };
    // A22 starts at A.data + (row_jump * newSize) + newSize
    MatrixView a22 = { A.data + (A.stride * newSize) + newSize, A.stride, newSize };

    MatrixView b11 = { B.data,                        B.stride, newSize };
    MatrixView b12 = { B.data + newSize,              B.stride, newSize };
    MatrixView b21 = { B.data + (B.stride * newSize), B.stride, newSize };
    MatrixView b22 = { B.data + (B.stride * newSize) + newSize, B.stride, newSize };

    // --- STRASSEN STEPS ---
    // We still have to allocate for the intermediate sums/differences, 
    // but we saved the allocations for the inputs!

    // M1 = (A11 + A22) * (B11 + B22)
    std::vector<float> t1 = add_views(a11, a22); 
    std::vector<float> t2 = add_views(b11, b22);
    Matrix m1 = strassen_recursive({t1.data(), newSize, newSize}, {t2.data(), newSize, newSize});

    // M2 = (A21 + A22) * B11
    std::vector<float> t3 = add_views(a21, a22);
    Matrix m2 = strassen_recursive({t3.data(), newSize, newSize}, b11);

    // M3 = A11 * (B12 - B22)
    std::vector<float> t4 = sub_views(b12, b22);
    Matrix m3 = strassen_recursive(a11, {t4.data(), newSize, newSize});

    // M4 = A22 * (B21 - B11)
    std::vector<float> t5 = sub_views(b21, b11);
    Matrix m4 = strassen_recursive(a22, {t5.data(), newSize, newSize});

    // M5 = (A11 + A12) * B22
    std::vector<float> t6 = add_views(a11, a12);
    Matrix m5 = strassen_recursive({t6.data(), newSize, newSize}, b22);

    // M6 = (A21 - A11) * (B11 + B12)
    std::vector<float> t7 = sub_views(a21, a11);
    std::vector<float> t8 = add_views(b11, b12);
    Matrix m6 = strassen_recursive({t7.data(), newSize, newSize}, {t8.data(), newSize, newSize});

    // M7 = (A12 - A22) * (B21 + B22)
    std::vector<float> t9 = sub_views(a12, a22);
    std::vector<float> t10 = add_views(b21, b22);
    Matrix m7 = strassen_recursive({t9.data(), newSize, newSize}, {t10.data(), newSize, newSize});

    // --- COMBINE RESULTS ---
    // Note: m1...m7 are currently std::vector<std::vector<float>> (Matrix)
    // because your base matrix_multiply_flatten returns that type.
    
    // You should ideally refactor your `addMatrices` to work on the return type of `strassen`.
    // Assuming m1...m7 are standard Matrix types:
    
    Matrix c11 = addMatrices(subtractMatrices(addMatrices(m1, m4), m5), m7);
    Matrix c12 = addMatrices(m3, m5);
    Matrix c21 = addMatrices(m2, m4);
    Matrix c22 = addMatrices(subtractMatrices(addMatrices(m1, m3), m2), m6);

    // Assemble final result
    Matrix result(n, std::vector<float>(n));
    for (size_t i = 0; i < newSize; i++) {
        for (size_t j = 0; j < newSize; j++) {
            result[i][j]                     = c11[i][j];
            result[i][j + newSize]           = c12[i][j];
            result[i + newSize][j]           = c21[i][j];
            result[i + newSize][j + newSize] = c22[i][j];
        }
    }
    return result;
}

// Wrapper to be called from Python
Matrix strassen_entry(const Matrix& mat1, const Matrix& mat2) {
    // 1. Flatten the initial inputs ONCE
    std::vector<float> f1 = flatten_to_vector(mat1); // You need to write this helper
    std::vector<float> f2 = flatten_to_vector(mat2);
    
    size_t n = mat1.size();
    
    // 2. Start recursion with Views
    return strassen_recursive({f1.data(), n, n}, {f2.data(), n, n});
}

std::vector<std::vector<float>> strassen_matrix_multiply(
	const std::vector<std::vector<float>>& mat1,
	const std::vector<std::vector<float>>& mat2){

	
	int n = mat1.size();

	if (n <= 2) {
		return matrix_multiply(mat1, mat2);
	}

	int newSize = n / 2;

	Matrix a11 = createMatrix(newSize);
	Matrix a12 = createMatrix(newSize);
	Matrix a21 = createMatrix(newSize);
	Matrix a22 = createMatrix(newSize);
	Matrix b11 = createMatrix(newSize);
	Matrix b12 = createMatrix(newSize);
	Matrix b21 = createMatrix(newSize);
	Matrix b22 = createMatrix(newSize);

	for (int i = 0; i < newSize; i++) {
		for (int j = 0; j < newSize; j++) {
			a11[i][j] = mat1[i][j];
			a12[i][j] = mat1[i][j + newSize];
			a21[i][j] = mat1[i + newSize][j];
			a22[i][j] = mat1[i + newSize][j + newSize];

			b11[i][j] = mat2[i][j];
			b12[i][j] = mat2[i][j + newSize];
			b21[i][j] = mat2[i + newSize][j];
			b22[i][j] = mat2[i + newSize][j + newSize];
		}
	}

	Matrix m1 = strassen_matrix_multiply(addMatrices(a11, a22), addMatrices(b11, b22));
	Matrix m2 = strassen_matrix_multiply(addMatrices(a21, a22), b11);
	Matrix m3 = strassen_matrix_multiply(a11, subtractMatrices(b12, b22));
	Matrix m4 = strassen_matrix_multiply(a22, subtractMatrices(b21, b11));
	Matrix m5 = strassen_matrix_multiply(addMatrices(a11, a12), b22);
	Matrix m6 = strassen_matrix_multiply(subtractMatrices(a21, a11), addMatrices(b11, b12));
	Matrix m7 = strassen_matrix_multiply(subtractMatrices(a12, a22), addMatrices(b21, b22));

	Matrix c11 = addMatrices(subtractMatrices(addMatrices(m1, m4), m5), m7);
	Matrix c12 = addMatrices(m3, m5);
	Matrix c21 = addMatrices(m2, m4);
	Matrix c22 = addMatrices(subtractMatrices(addMatrices(m1, m3), m2), m6);

	Matrix result = createMatrix(n);

	for (int i = 0; i < newSize; i++) {
		for (int j = 0; j < newSize; j++) {
			result[i][j] = c11[i][j];
			result[i][j + newSize] = c12[i][j];
			result[i + newSize][j] = c21[i][j];
			result[i + newSize][j + newSize] = c22[i][j];
		}
	}
	return result;

}

void increment(int& value){

	value++;
}

float multiply(float& val1, float& val2){

	return val1 * val2;
}

float divide(float& val1, float& val2){

	return val1 / val2;
}

float add(float& val1, float& val2){

	return val1 + val2;
}



PYBIND11_MODULE(calculator_api, m, py::mod_gil_not_used()) {
    m.doc() = "calculator api"; 

    m.def("add", &add, "A function that adds two numbers");
    m.def("multiply", &multiply, "A function that multiplies two numbers");
    m.def("divide", &divide, "A function that divides two numbers");
	m.def("matrix_multiply", &matrix_multiply, "A function that multiplies two matrices");

	m.def("strassen_matrix_multiply", &strassen_matrix_multiply, "A function that multiplies two matrices using Strassen's algorithm");
	m.def("increment", &increment, "A function that increments a number");
	m.def("createMatrix", &createMatrix, "A function that creates an n x n matrix initialized to zero");
	m.def("addMatrices", &addMatrices, "A function that adds two matrices");
	m.def("subtractMatrices", &subtractMatrices, "A function that subtracts two matrices");

	m.def("matrix_multiply_flatten", &matrix_multiply_flatten, 
      "A function that multiplies two flattened matrices");
	py::class_<matrix2d>(m, "matrix2d")
    .def(py::init<std::vector<float>, size_t, size_t>()) // You'd need a constructor
    .def_readwrite("v_", &matrix2d::v_)
    .def_readwrite("x_", &matrix2d::x_)
    .def_readwrite("y_", &matrix2d::y_);


	m.def("strassen_entry", &strassen_entry, "A function that multiplies two matrices using Strassen's algorithm with views");


}

