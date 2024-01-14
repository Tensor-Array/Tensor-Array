/*
Copyright 2024 TensorArray-Creators

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <limits>
#include <cassert>
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif

#define USING_DATA_TYPE_NVIDIA_FLOAT_8 (__nv_fp8_e5m2)(__nv_fp8_e4m3)
#define USING_DATA_TYPE_NVIDIA_FLOAT (__half)(__nv_bfloat16)
#define USING_DATA_TYPE_FLOAT (float)(double)
#define USING_DATA_TYPE_SINT (int8_t)(int16_t)(int32_t)(int64_t)
#define USING_DATA_TYPE_UINT (uint8_t)(uint16_t)(uint32_t)(uint64_t)
#define USING_DATA_TYPE USING_DATA_TYPE_SINT USING_DATA_TYPE_UINT USING_DATA_TYPE_FLOAT

#define LOOP(seq) END(A seq)
#define BODY(x) ADD_CODE(x)
#define A(x) BODY(x) B
#define B(x) BODY(x) A
#define A_END
#define B_END
#define END(...) END_(__VA_ARGS__)
#define END_(...) __VA_ARGS__##_END


namespace tensor_array
{
    namespace value
    {
        using namespace devices;
        template <typename T, unsigned int blockSize>
        __device__ void warp_reduce_sum(volatile T *sdata, unsigned int tid)
        {
            if constexpr (blockSize >= 1024)
                sdata[tid] += sdata[tid + 512];
            if constexpr (blockSize >= 512)
                sdata[tid] += sdata[tid + 256];
            if constexpr (blockSize >= 256)
                sdata[tid] += sdata[tid + 128];
            if constexpr (blockSize >= 128)
                sdata[tid] += sdata[tid + 64];
            if constexpr (blockSize >= 64)
                sdata[tid] += sdata[tid + 32];
            if constexpr (blockSize >= 32)
                sdata[tid] += sdata[tid + 16];
            if constexpr (blockSize >= 16)
                sdata[tid] += sdata[tid + 8];
            if constexpr (blockSize >= 8)
                sdata[tid] += sdata[tid + 4];
            if constexpr (blockSize >= 4)
                sdata[tid] += sdata[tid + 2];
            if constexpr (blockSize >= 2)
                sdata[tid] += sdata[tid + 1];
        }

        template <typename T, unsigned int blockSize>
        __device__ void warp_reduce_max(volatile T *sdata, volatile unsigned int *sindex, unsigned int tid)
        {
            if constexpr (blockSize >= 1024)
                if (sdata[tid] < sdata[tid + 512])
                {
                    sdata[tid] = sdata[tid + 512];
                    sindex[tid] = sindex[tid + 512];
                }
            if constexpr (blockSize >= 512)
                if (sdata[tid] < sdata[tid + 256])
                {
                    sdata[tid] = sdata[tid + 256];
                    sindex[tid] = sindex[tid + 256];
                }
            if constexpr (blockSize >= 256)
                if (sdata[tid] < sdata[tid + 128])
                {
                    sdata[tid] = sdata[tid + 128];
                    sindex[tid] = sindex[tid + 128];
                }
            if constexpr (blockSize >= 128)
                if (sdata[tid] < sdata[tid + 64])
                {
                    sdata[tid] = sdata[tid + 64];
                    sindex[tid] = sindex[tid + 64];
                }
            if constexpr (blockSize >= 64)
                if (sdata[tid] < sdata[tid + 32])
                {
                    sdata[tid] = sdata[tid + 32];
                    sindex[tid] = sindex[tid + 32];
                }
            if constexpr (blockSize >= 32)
                if (sdata[tid] < sdata[tid + 16])
                {
                    sdata[tid] = sdata[tid + 16];
                    sindex[tid] = sindex[tid + 16];
                }
            if constexpr (blockSize >= 16)
                if (sdata[tid] < sdata[tid + 8])
                {
                    sdata[tid] = sdata[tid + 8];
                    sindex[tid] = sindex[tid + 8];
                }
            if constexpr (blockSize >= 8)
                if (sdata[tid] < sdata[tid + 4])
                {
                    sdata[tid] = sdata[tid + 4];
                    sindex[tid] = sindex[tid + 4];
                }
            if constexpr (blockSize >= 4)
                if (sdata[tid] < sdata[tid + 2])
                {
                    sdata[tid] = sdata[tid + 2];
                    sindex[tid] = sindex[tid + 2];
                }
            if constexpr (blockSize >= 2)
                if (sdata[tid] < sdata[tid + 1])
                {
                    sdata[tid] = sdata[tid + 1];
                    sindex[tid] = sindex[tid + 1];
                }
        }

        template <typename T, unsigned int blockSize>
        __device__ void warp_reduce_min(volatile T *sdata, volatile unsigned int *sindex, unsigned int tid)
        {
            if constexpr (blockSize >= 1024)
                if (sdata[tid] > sdata[tid + 512])
                {
                    sdata[tid] = sdata[tid + 512];
                    sindex[tid] = sindex[tid + 512];
                }
            if constexpr (blockSize >= 512)
                if (sdata[tid] > sdata[tid + 256])
                {
                    sdata[tid] = sdata[tid + 256];
                    sindex[tid] = sindex[tid + 256];
                }
            if constexpr (blockSize >= 256)
                if (sdata[tid] > sdata[tid + 128])
                {
                    sdata[tid] = sdata[tid + 128];
                    sindex[tid] = sindex[tid + 128];
                }
            if constexpr (blockSize >= 128)
                if (sdata[tid] > sdata[tid + 64])
                {
                    sdata[tid] = sdata[tid + 64];
                    sindex[tid] = sindex[tid + 64];
                }
            if constexpr (blockSize >= 64)
                if (sdata[tid] > sdata[tid + 32])
                {
                    sdata[tid] = sdata[tid + 32];
                    sindex[tid] = sindex[tid + 32];
                }
            if constexpr (blockSize >= 32)
                if (sdata[tid] > sdata[tid + 16])
                {
                    sdata[tid] = sdata[tid + 16];
                    sindex[tid] = sindex[tid + 16];
                }
            if constexpr (blockSize >= 16)
                if (sdata[tid] > sdata[tid + 8])
                {
                    sdata[tid] = sdata[tid + 8];
                    sindex[tid] = sindex[tid + 8];
                }
            if constexpr (blockSize >= 8)
                if (sdata[tid] > sdata[tid + 4])
                {
                    sdata[tid] = sdata[tid + 4];
                    sindex[tid] = sindex[tid + 4];
                }
            if constexpr (blockSize >= 4)
                if (sdata[tid] > sdata[tid + 2])
                {
                    sdata[tid] = sdata[tid + 2];
                    sindex[tid] = sindex[tid + 2];
                }
            if constexpr (blockSize >= 2)
                if (sdata[tid] > sdata[tid + 1])
                {
                    sdata[tid] = sdata[tid + 1];
                    sindex[tid] = sindex[tid + 1];
                }
        }

        template <typename T, unsigned int blockSize>
        __global__ void array_reduce_sum(T *g_odata, const T *g_idata, unsigned int n)
        {
            __shared__ T sdata[blockSize];
            unsigned int tid = threadIdx.x;
            unsigned int gridSize = blockDim.x * gridDim.x;
            sdata[tid] = 0;
            if (blockIdx.x * blockDim.x + tid < n)
                sdata[tid] += g_idata[blockIdx.x * blockDim.x + tid];
            __syncthreads();
            if (tid < 512)
                warp_reduce_sum<T, blockSize>(sdata, tid);
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }

        template <typename T, unsigned int blockSize>
        __global__ void array_reduce_max(T *g_odata, unsigned int *g_oindex, const T *g_idata, unsigned int n)
        {
            __shared__ T sdata[blockSize];
            __shared__ unsigned int sindex[blockSize];
            unsigned int tid = threadIdx.x;
            unsigned int gridSize = blockDim.x * gridDim.x;
            sdata[tid] = std::numeric_limits<T>::min();
            if (blockIdx.x * blockDim.x + tid < n)
                sdata[tid] = g_idata[blockIdx.x * blockDim.x + tid];
            __syncthreads();
            if (tid < 512)
                warp_reduce_max<T, blockSize>(sdata, sindex, tid);
            if (tid == 0)
            {
                g_odata[blockIdx.x] = sdata[0];
                g_oindex[blockIdx.x] = sindex[0];
            }
        }

        template <typename T, unsigned int blockSize>
        __global__ void array_reduce_min(T *g_odata, unsigned int *g_oindex, const T *g_idata, unsigned int n)
        {
            __shared__ T sdata[blockSize];
            __shared__ unsigned int sindex[blockSize];
            unsigned int tid = threadIdx.x;
            unsigned int gridSize = blockDim.x * gridDim.x;
            sdata[tid] = std::numeric_limits<T>::max();
            if (blockIdx.x * blockDim.x + tid < n)
                sdata[tid] = g_idata[blockIdx.x * blockDim.x + tid];
            __syncthreads();
            if (tid < 512)
                warp_reduce_min<T, blockSize>(sdata, sindex, tid);
            if (tid == 0)
            {
                g_odata[blockIdx.x] = sdata[0];
                g_oindex[blockIdx.x] = sindex[0];
            }
        }

        bool equal_dim_size(const TensorBase&, const TensorBase&);
        Tensor multiply(const Tensor&, const Tensor&, bool, const DataBuffer&);

        Tensor reduce_sum(const Tensor& a)
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			temp.push_back(std::make_pair(a, Derivation(values(a.get_buffer().shape(), 1).tensor_cast(a.get_buffer().type(), false), multiply)));
			cudaError_t cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			const TensorBase& base_a = a.get_buffer();
			cuda_status = cudaMalloc(&c_ptr, base_a.data_size());
            device_memcpy(&c_ptr, this_cuda, base_a.data(), base_a.get_device(), base_a.data_size());
            std::vector<unsigned int> shape_c = a.get_buffer().shape();
            std::size_t c_size = a.get_buffer().data_size() / get_sizeof_type(base_a.type());
            constexpr unsigned int thread_value = 1024U;
			dim3 block_dim(shape_c[shape_c.size() - 1]);
			dim3 grid_dim(c_size / block_dim.x);
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE)) \
{ \
array_reduce_sum<TYPE, thread_value><<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(c_ptr), c_size); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), a.get_buffer().shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf));
		}

        Tensor reduce_max(const Tensor& a)
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			temp.push_back(std::make_pair(a, Derivation(values(a.get_buffer().shape(), 1).tensor_cast(a.get_buffer().type(), false), multiply)));
			cudaError_t cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			const TensorBase& base_a = a.get_buffer();
			cuda_status = cudaMalloc(&c_ptr, base_a.data_size());
            device_memcpy(&c_ptr, this_cuda, base_a.data(), base_a.get_device(), base_a.data_size());
            std::vector<unsigned int> shape_c = a.get_buffer().shape();
            std::size_t c_size = a.get_buffer().data_size() / get_sizeof_type(base_a.type());
            constexpr unsigned int thread_value = 1024U;
			dim3 block_dim(shape_c[shape_c.size() - 1]);
			dim3 grid_dim(c_size / block_dim.x);
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE)) \
{ \
array_reduce_sum<TYPE, thread_value><<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(c_ptr), c_size); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), a.get_buffer().shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf));
		}

        Tensor reduce_min(const Tensor& a)
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
            temp.push_back(std::make_pair(a, Derivation(values(a.get_buffer().shape(), 1).tensor_cast(a.get_buffer().type(), false), multiply)));
			cudaError_t cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			const TensorBase& base_a = a.get_buffer();
			cuda_status = cudaMalloc(&c_ptr, base_a.data_size());
            device_memcpy(&c_ptr, this_cuda, base_a.data(), base_a.get_device(), base_a.data_size());
            std::vector<unsigned int> shape_c = a.get_buffer().shape();
            std::size_t c_size = a.get_buffer().data_size() / get_sizeof_type(base_a.type());
            constexpr unsigned int thread_value = 1024U;
			dim3 block_dim(shape_c[shape_c.size() - 1]);
			dim3 grid_dim(c_size / block_dim.x);
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE)) \
{ \
array_reduce_sum<TYPE, thread_value><<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(c_ptr), c_size); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), a.get_buffer().shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf));
		}
    }
}
