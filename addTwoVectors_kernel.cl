__kernel void vecAdd(__global float* destination, __global float* a, __global float* b)
{
    int gid = get_global_id(0);
 
   destination[gid] = a[gid] + b[gid];
}