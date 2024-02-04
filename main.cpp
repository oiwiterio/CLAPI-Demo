#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int main() {
    // 矩阵大小
    const int N = 5;
    float A[N * N] = { 
        -0.61016697,0.1699776,-0.23161632,-0.83269614,0.36900806,
        -0.86159897,0.34153128,0.0055288672,0.6487526,-0.8452286,
        -0.29620767,0.38445604,-0.9451324,0.8718862,0.8791129,
        -0.588762,0.88659006,-0.5369606,0.42085797,0.7559038,
        -0.15419173,-0.7793449,0.32992667,-0.63829416,-0.77858704
    }; // 初始化矩阵A
    float B[N * N] = {
        0.40003085,-0.06757355,-0.24704313,-0.57165056,-0.7554205,
        -0.57865614,0.9707482,0.22802413,0.11706495,0.17188013,
        0.47308856,0.117473245,0.20759785,0.7907691,0.5919476,
        -0.6746741,-0.90659714,0.08401322,-0.565932,0.3789456,
        -0.39856273,-0.9015139,0.96552867,0.76055574,-0.77838176,
    }; // 初始化矩阵B
    float C[N * N]; // 结果矩阵

    // 加载OpenCL内核
    FILE* fp;
    char fileName[] = "./matadd.cl";
    char* source_str;
    size_t source_size;


    errno_t err = fopen_s(&fp, fileName, "r");
    if (err != 0) {
        fprintf(stderr, "未能成功得到内核源码\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // 获取平台和设备信息
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // 创建OpenCL上下文
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // 创建命令队列
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0 // 0作为终结符
    };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);


    // 创建内存缓冲
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &ret);

    // 复制列表到内存缓冲
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, N * N * sizeof(float), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, N * N * sizeof(float), B, 0, NULL, NULL);

    // 准备并运行内核程序
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // 创建OpenCL内核
    cl_kernel kernel = clCreateKernel(program, "matrix_add", &ret);

    // 设置OpenCL内核参数
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&N);

    // 执行OpenCL内核
    size_t global_item_size[2] = { N, N };
    size_t local_item_size[2] = { 1, 1 };
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

    // 读取结果
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, N * N * sizeof(float), C, 0, NULL, NULL);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i * N + j]);
        }
        printf("\n\n");
    }

    printf("加上\n\n");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", B[i * N + j]);
        }
        printf("\n\n");
    }

    printf("得到\n\n");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f + %f   ", A[i * N + j], B[i * N + j]);
        }
        printf("\n\n");
    }

    printf("等于\n\n");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n\n");
    }


    // 清理
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(source_str);


    getchar();
    return 0;
}
/*
-0.610167 0.169978 -0.231616 -0.832696 0.369008

-0.861599 0.341531 0.005529 0.648753 -0.845229

-0.296208 0.384456 -0.945132 0.871886 0.879113

-0.588762 0.886590 -0.536961 0.420858 0.755904

-0.154192 -0.779345 0.329927 -0.638294 -0.778587

加上

0.400031 -0.067574 -0.247043 -0.571651 -0.755421

-0.578656 0.970748 0.228024 0.117065 0.171880

0.473089 0.117473 0.207598 0.790769 0.591948

-0.674674 -0.906597 0.084013 -0.565932 0.378946

-0.398563 -0.901514 0.965529 0.760556 -0.778382

得到

-0.610167 + 0.400031   0.169978 + -0.067574   -0.231616 + -0.247043   -0.832696 + -0.571651   0.369008 + -0.755421

-0.861599 + -0.578656   0.341531 + 0.970748   0.005529 + 0.228024   0.648753 + 0.117065   -0.845229 + 0.171880

-0.296208 + 0.473089   0.384456 + 0.117473   -0.945132 + 0.207598   0.871886 + 0.790769   0.879113 + 0.591948

-0.588762 + -0.674674   0.886590 + -0.906597   -0.536961 + 0.084013   0.420858 + -0.565932   0.755904 + 0.378946

-0.154192 + -0.398563   -0.779345 + -0.901514   0.329927 + 0.965529   -0.638294 + 0.760556   -0.778587 + -0.778382

等于

-0.210136 0.102404 -0.478659 -1.404347 -0.386412

-1.440255 1.312279 0.233553 0.765818 -0.673348

0.176881 0.501929 -0.737535 1.662655 1.471061

-1.263436 -0.020007 -0.452947 -0.145074 1.134849

-0.552754 -1.680859 1.295455 0.122262 -1.556969
*/