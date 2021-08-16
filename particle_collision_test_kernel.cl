__kernel void vecAdd(__global float* destination, __global float* a)
{

    int gid_0 = get_global_id(0);
    int gid_1 = get_global_id(1);

    printf("gid_0: %i, gid_1: %i\n", gid_0, gid_1);

   destination[gid_0] = 1;
}