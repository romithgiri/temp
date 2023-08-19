import tensorflow as tf

def matrix_multiplication(A, B):
  """
  Performs matrix multiplication of two matrices.

  Args:
    A: The first matrix.
    B: The second matrix.

  Returns:
    The product of the two matrices.
  """
  return tf.matmul(A, B)

def find_eigen_values_and_vectors(A):
  """
  Finds the Eigen values and vectors of a matrix.

  Args:
    A: The matrix to find the Eigen values and vectors for.

      Returns:
    A tuple of the Eigen values and vectors.
  """
  eigen_values, eigen_vectors = tf.linalg.eig(A)
  return eigen_values, eigen_vectors

if __name__ == "__main__":
  # Create two matrices.
  A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
  B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

  # Perform matrix multiplication.
  C = matrix_multiplication(A, B)

  # Find the Eigen values and vectors of matrix A.
  eigen_values, eigen_vectors = find_eigen_values_and_vectors(A)

  # Print the results.
  print("Matrix A:")
  print(A)
  print()
  print("Matrix B:")
  print(B)
  print()
  print("Matrix C (A * B):")
  print(C)
  print()
  print("Eigen values of matrix A:")
  print(eigen_values)
  print()
  print("Eigen vectors of matrix A:")
  print(eigen_vectors)
