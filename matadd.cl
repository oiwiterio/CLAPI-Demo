__kernel void matrix_add(__global float* A, __global float* B, __global float* C, const unsigned int N) {
    int row = get_global_id(0); // 获取全局ID
    int col = get_global_id(1);

    int index = row * N + col;
    C[index] = A[index] + B[index];
}