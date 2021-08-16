#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (16)//suppose we have a vector with 128 elements
#define MAX_SOURCE_SIZE (0x100000)

//USING: https://www.sharcnet.ca/help/index.php/Porting_CUDA_to_OpenCL

int main()
{
	//Platform we're using
	cl_platform_id platform_id = NULL;

	//Device we're using
	cl_device_id device_id = NULL;

	//Current context
	cl_context context = NULL;

	//Command Queue
	cl_command_queue command_queue = NULL;

	//Device Memory
	cl_mem memobj_pVals;

	cl_mem memobj_corrections;

	//Program executable created from source
	cl_program program = NULL;

	cl_kernel kernel = NULL;

	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;

	cl_int ret; //accepts return values for APIs

	//Memory arrays - essentially arrays of floats
	float mem_pVals[MEM_SIZE];

	float mem_corrections[MEM_SIZE];

	FILE* fp;
	const char fileName[] = "./particle_collision_test_kernel.cl";
	size_t source_size;
	char* source_str;
	cl_int i;

	//Open the kernel file

	fp = fopen(fileName, "r");

	if (!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n"); exit(1);
	}

	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//Initialize memory
	for (i = 0; i < MEM_SIZE; i++) { mem_pVals[i] = i; }

	//Get device info
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	//Create a context on the device
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	//Create command queue (stream)
	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

	//Allocate read/write memory on the device
	memobj_pVals = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);

	memobj_corrections = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);

	//Copy memory from host memory to device memory (Float arrays to cl_mem objects)
	//(CL_TRUE = block read/write)
	ret = clEnqueueWriteBuffer(command_queue, memobj_pVals, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem_pVals, 0, NULL, NULL);

	//ret = clEnqueueWriteBuffer(command_queue, memobj_result, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem_result, 0, NULL, NULL);

	//Create program for this context - load source code specified by text strings into program object
	program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

	//Build a program executable from program source (or binary)
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	kernel = clCreateKernel(program, "vecAdd", &ret);

	//Set argument value for a specific arguent of a kernel

	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_corrections);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_pVals);

	//Define global size and local size
	size_t global_work_size[2] = { MEM_SIZE, MEM_SIZE};
	size_t local_work_size[2] = { MEM_SIZE/2, MEM_SIZE/2};

	//Enqueue a command to execute a kernel on a device ("1" = 1 dimensional work)
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, (size_t*)&global_work_size, local_work_size, 0, NULL, NULL);

	//Copy memory from device to host
	ret = clEnqueueReadBuffer(command_queue, memobj_corrections, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem_corrections, 0, NULL, NULL);

	//Print out result
	//for (i = 0; i < MEM_SIZE; i++) printf("Memory array mem[%d] : %.2f\n", i, mem_corrections[i]);

	printf("\nopencl_test\n");

	//clFlush only guarantees that all queued commands to command_queue get issued to the appropriate device
	//There is no guarantee that they will be complete after clFlush returns
	ret = clFlush(command_queue);
	//clFinish blocks until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed.
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(memobj_pVals);//free memory on device
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(source_str);//free memory on host

	return 0;

}