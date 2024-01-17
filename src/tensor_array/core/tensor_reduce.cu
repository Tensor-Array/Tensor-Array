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

        template <typename T, unsigned int BatchBlockSize, unsigned int DimBlockSize, unsigned int ContentBlockSize>
        __device__ void warp_reduce_sum(T (*sdata)[BatchBlockSize][DimBlockSize][ContentBlockSize], unsigned int value)
        {
            (*sdata)[threadIdx.x][threadIdx.z][threadIdx.y] += (*sdata)[threadIdx.x][threadIdx.z + value][threadIdx.y];
        }

        template <typename T, unsigned int BatchBlockSize, unsigned int DimBlockSize, unsigned int ContentBlockSize, typename ... Args>
        __device__ void warp_reduce_functions
        (
            void(*func)(T (*)[BatchBlockSize][DimBlockSize][ContentBlockSize], unsigned int),
            T (*sdata)[BatchBlockSize][DimBlockSize][ContentBlockSize],
            Args ... args
        )
        {
            if constexpr (DimBlockSize >= 1024) if (threadIdx.z < 512) func(sdata, 512, args...);
            if constexpr (DimBlockSize >= 512) if (threadIdx.z < 256) func(sdata, 256, args...);
            if constexpr (DimBlockSize >= 256) if (threadIdx.z < 128) func(sdata, 128, args...);
            if constexpr (DimBlockSize >= 128) if (threadIdx.z < 64) func(sdata, 64, args...);
            if constexpr (DimBlockSize >= 64) if (threadIdx.z < 32) func(sdata, 32, args...);
            if constexpr (DimBlockSize >= 32) if (threadIdx.z < 16) func(sdata, 16, args...);
            if constexpr (DimBlockSize >= 16) if (threadIdx.z < 8) func(sdata, 8, args...);
            if constexpr (DimBlockSize >= 8) if (threadIdx.z < 4) func(sdata, 4, args...);
            if constexpr (DimBlockSize >= 4) if (threadIdx.z < 2) func(sdata, 2, args...);
            if constexpr (DimBlockSize >= 2) if (threadIdx.z < 1) func(sdata, 1, args...);
        }

        template <typename T, unsigned int BatchBlockSize, unsigned int DimBlockSize, unsigned int ContentBlockSize>
        __device__ void warp_reduce_max(T (*sdata)[BatchBlockSize][DimBlockSize][ContentBlockSize], unsigned int value, unsigned int (*sindex)[BatchBlockSize][DimBlockSize][ContentBlockSize])
        {
            if (sdata[threadIdx.x][threadIdx.z][threadIdx.y] < sdata[threadIdx.x][threadIdx.z + value][threadIdx.y])
            {
                sdata[threadIdx.x][threadIdx.z + value][threadIdx.y] = sdata[threadIdx.x][threadIdx.z + value][threadIdx.y];
                sindex[threadIdx.x][threadIdx.z + value][threadIdx.y] = sindex[threadIdx.x][threadIdx.z + value][threadIdx.y];
            }
        }

        template <typename T, unsigned int BatchBlockSize, unsigned int DimBlockSize, unsigned int ContentBlockSize>
        __device__ void warp_reduce_min(T (*sdata)[BatchBlockSize][DimBlockSize][ContentBlockSize], unsigned int value, unsigned int (*sindex)[BatchBlockSize][DimBlockSize][ContentBlockSize])
        {
            if (sdata[threadIdx.x][threadIdx.z][threadIdx.y] > sdata[threadIdx.x][threadIdx.z + value][threadIdx.y])
            {
                sdata[threadIdx.x][threadIdx.z][threadIdx.y] = sdata[threadIdx.x][threadIdx.z + value][threadIdx.y];
                sindex[threadIdx.x][threadIdx.z][threadIdx.y] = sindex[threadIdx.x][threadIdx.z + value][threadIdx.y];
            }
        }

        template <typename T, unsigned int BatchBlockSize, unsigned int BlockSize, unsigned int ContentBlockSize>
        __global__ void array_reduce_sum(T *g_odata, const T *g_idata, unsigned int batch_size, unsigned int n, unsigned int content_size)
        {
            __shared__ T sdata[BatchBlockSize][BlockSize][ContentBlockSize];
            unsigned int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int content_id = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int tid = threadIdx.z;
            unsigned int gridSize = blockDim.z * gridDim.z;
            sdata[threadIdx.x][threadIdx.z][threadIdx.y] = 0;
            if (batch_id < batch_size && blockIdx.z * blockDim.z + tid < n && content_id < content_size)
                sdata[threadIdx.x][threadIdx.z][threadIdx.y] +=
                g_idata
                [
                    batch_id * n * content_size +
                    tid * content_size +
                    content_id
                ];
            __syncthreads();
            if (tid < 512)
                warp_reduce_functions(&warp_reduce_sum, &sdata);
            if (tid == 0)
                g_odata[
                    batch_id * blockDim.z * content_size +
                    tid * content_size +
                    content_id
                ] = sdata[threadIdx.x][threadIdx.z][threadIdx.y];
        }

        template <typename T, unsigned int BatchBlockSize, unsigned int BlockSize, unsigned int ContentBlockSize>
        __global__ void array_reduce_max(T *g_odata, unsigned int *g_oindex, const T *g_idata, unsigned int batch_size, unsigned int n, unsigned int content_size)
        {
            __shared__ T sdata[BatchBlockSize][BlockSize][ContentBlockSize];
            __shared__ unsigned int sindex[BatchBlockSize][BlockSize][ContentBlockSize];
            unsigned int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int content_id = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int tid = threadIdx.z;
            unsigned int gridSize = blockDim.z * gridDim.z;
            sdata[threadIdx.x][threadIdx.z][threadIdx.y] = -std::numeric_limits<T>::infinity();
            sindex[threadIdx.x][threadIdx.z][threadIdx.y] = threadIdx.z;
            if (batch_id < batch_size && blockIdx.z * blockDim.z + tid < n && content_id < content_size)
                sdata[threadIdx.x][threadIdx.z][threadIdx.y] =
                g_idata
                [
                    batch_id * n * content_size +
                    tid * content_size +
                    content_id
                ];
            __syncthreads();
            if (tid < 512)
                warp_reduce_functions(&warp_reduce_max, &sdata, &sindex);
            if (tid == 0)
            {
                g_odata[
                    batch_id * blockDim.z * content_size +
                    blockIdx.z * content_size +
                    content_id
                ] = sdata[threadIdx.x][threadIdx.z][threadIdx.y];
                g_oindex[
                    batch_id * blockDim.z * content_size +
                    blockIdx.z * content_size +
                    content_id
                ] = sindex[threadIdx.x][threadIdx.z][threadIdx.y];
            }
        }

        template <typename T, unsigned int BatchBlockSize, unsigned int BlockSize, unsigned int ContentBlockSize>
        __global__ void array_reduce_min(T *g_odata, unsigned int *g_oindex, const T *g_idata, unsigned int batch_size, unsigned int n, unsigned int content_size)
        {
            __shared__ T sdata[BatchBlockSize][BlockSize][ContentBlockSize];
            __shared__ unsigned int sindex[BatchBlockSize][BlockSize][ContentBlockSize];
            unsigned int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int content_id = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int tid = threadIdx.z;
            unsigned int gridSize = blockDim.z * gridDim.z;
            sdata[threadIdx.x][threadIdx.z][threadIdx.y] = std::numeric_limits<T>::infinity();
            sindex[threadIdx.x][threadIdx.z][threadIdx.y] = threadIdx.z;
            if (batch_id < batch_size && blockIdx.z * blockDim.z + tid < n && content_id < content_size)
                sdata[threadIdx.x][threadIdx.z][threadIdx.y] =
                g_idata
                [
                    batch_id * n * content_size +
                    tid * content_size +
                    content_id
                ];
            __syncthreads();
            if (tid < 512)
                warp_reduce_functions(&warp_reduce_min, &sdata, &sindex);
            if (tid == 0)
            {
                g_odata[
                    batch_id * blockDim.z * content_size +
                    blockIdx.z * content_size +
                    content_id
                ] = sdata[threadIdx.x][threadIdx.z][threadIdx.y];
                g_oindex[
                    batch_id * blockDim.z * content_size +
                    blockIdx.z * content_size +
                    content_id
                ] = sindex[threadIdx.x][threadIdx.z][threadIdx.y];
            }
        }

        bool equal_dim_size(const TensorBase&, const TensorBase&);
        Tensor derive_reduce_sum(const Tensor& a, const Tensor& b, bool is_derive, const DataBuffer& databuf)
        {
            return multiply(a, b, is_derive, databuf);
        }

        Tensor Tensor::reduce_sum(unsigned char dim) const
		{
            std::vector<unsigned int> shape_c = this->get_buffer().shape();
            assert(dim < shape_c.size());
			std::vector<std::pair<Tensor, Derivation>> temp;
			temp.push_back(std::make_pair(*this, Derivation(values(shape_c, 1).tensor_cast(this->get_buffer().type(), false), derive_reduce_sum)));
			cudaError_t cuda_status;
			TensorBase other_buf;

			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			const TensorBase& base_a = this->get_buffer();
			cuda_status = cudaMalloc(&c_ptr, base_a.data_size());
            device_memcpy(&c_ptr, this_cuda, base_a.data(), base_a.get_device(), base_a.data_size());

            unsigned int dim_x = 1;
            for (unsigned char i = 0; i < dim; i++)
                dim_x *= shape_c[i];

            unsigned int dim_y = 1;
            for (unsigned char i = dim+1; i < shape_c.size(); i++)
                dim_y *= shape_c[i];
            
            constexpr unsigned int thread_value_x = 8U;
            constexpr unsigned int thread_value_y = 16U;
            constexpr unsigned int thread_value_z = 8U;
			dim3 block_dim(thread_value_x, thread_value_y, thread_value_z);
			dim3 grid_dim
            (
                dim_x / block_dim.x + (dim_x % block_dim.x  ? 1U : 0U),
                dim_y / block_dim.y + (dim_y % block_dim.y ? 1U : 0U),
                shape_c[dim] / block_dim.z + (shape_c[dim] % block_dim.z ? 1U : 0U)
            );
#define ADD_CODE(TYPE) \
if(base_a.type() == typeid(TYPE)) \
{ \
while (shape_c[dim] > 1) \
{ \
array_reduce_sum<TYPE, thread_value_x, thread_value_z, thread_value_y><<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(c_ptr), dim_x, shape_c[dim], dim_y); \
cuda_status = cudaDeviceSynchronize(); \
shape_c[dim] = grid_dim.z; \
grid_dim.z = grid_dim.z / block_dim.z  + (grid_dim.z % block_dim.z ? 1U : 0U); \
} \
other_buf = TensorBase(typeid(TYPE), shape_c, c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf));
		}

        Tensor Tensor::reduce_max(unsigned char dim) const
		{
            std::vector<unsigned int> shape_c = this->get_buffer().shape();
            assert(dim < shape_c.size());
			cudaError_t cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			const TensorBase& base_a = this->get_buffer();
			cuda_status = cudaMalloc(&c_ptr, base_a.data_size());
            device_memcpy(&c_ptr, this_cuda, base_a.data(), base_a.get_device(), base_a.data_size());

            unsigned int dim_x = 1;
            for (unsigned char i = 0; i < dim; i++)
                dim_x *= shape_c[i];

            unsigned int dim_y = 1;
            for (unsigned char i = dim+1; i < shape_c.size(); i++)
                dim_y *= shape_c[i];
            
            constexpr unsigned int thread_value_x = 8U;
            constexpr unsigned int thread_value_y = 16U;
            constexpr unsigned int thread_value_z = 8U;
			dim3 block_dim(thread_value_x, thread_value_y, thread_value_z);
			dim3 grid_dim
            (
                dim_x / block_dim.x + (dim_x % block_dim.x  ? 1U : 0U),
                dim_y / block_dim.y + (dim_y % block_dim.y ? 1U : 0U),
                shape_c[dim] / block_dim.z + (shape_c[dim] % block_dim.z ? 1U : 0U)
            );
#define ADD_CODE(TYPE) \
if(base_a.type() == typeid(TYPE)) \
{ \
while (shape_c[dim] > 1) \
{ \
array_reduce_sum<TYPE, thread_value_x, thread_value_z, thread_value_y><<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(c_ptr), dim_x, shape_c[dim], dim_y); \
cuda_status = cudaDeviceSynchronize(); \
shape_c[dim] = grid_dim.z; \
grid_dim.z = grid_dim.z / block_dim.z  + (grid_dim.z % block_dim.z ? 1U : 0U); \
} \
other_buf = TensorBase(typeid(TYPE), base_a.shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
            std::vector<std::pair<Tensor, Derivation>> temp;
			temp.push_back(std::make_pair(*this, Derivation(values(this->get_buffer().shape(), 1).tensor_cast(this->get_buffer().type(), false), derive_reduce_sum)));
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf));
		}

        Tensor Tensor::reduce_min(unsigned char dim) const
		{
            std::vector<unsigned int> shape_c = this->get_buffer().shape();
            assert(dim < shape_c.size());
			cudaError_t cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			const TensorBase& base_a = this->get_buffer();
			cuda_status = cudaMalloc(&c_ptr, base_a.data_size());
            device_memcpy(&c_ptr, this_cuda, base_a.data(), base_a.get_device(), base_a.data_size());
            unsigned int dim_x = 1;
            for (unsigned char i = 0; i < dim; i++)
                dim_x *= shape_c[i];

            unsigned int dim_y = 1;
            for (unsigned char i = dim+1; i < shape_c.size(); i++)
                dim_y *= shape_c[i];
            
            constexpr unsigned int thread_value_x = 8U;
            constexpr unsigned int thread_value_y = 16U;
            constexpr unsigned int thread_value_z = 8U;
			dim3 block_dim(thread_value_x, thread_value_y, thread_value_z);
			dim3 grid_dim
            (
                dim_x / block_dim.x + (dim_x % block_dim.x  ? 1U : 0U),
                dim_y / block_dim.y + (dim_y % block_dim.y ? 1U : 0U),
                shape_c[dim] / block_dim.z + (shape_c[dim] % block_dim.z ? 1U : 0U)
            );
#define ADD_CODE(TYPE) \
if(base_a.type() == typeid(TYPE)) \
{ \
while (shape_c[dim] > 1) \
{ \
array_reduce_sum<TYPE, thread_value_x, thread_value_z, thread_value_y><<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(c_ptr), dim_x, shape_c[dim], dim_y); \
cuda_status = cudaDeviceSynchronize(); \
shape_c[dim] = grid_dim.z; \
grid_dim.z = grid_dim.z / block_dim.z  + (grid_dim.z % block_dim.z ? 1U : 0U); \
} \
other_buf = TensorBase(typeid(TYPE), base_a.shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
            std::vector<std::pair<Tensor, Derivation>> temp;
            temp.push_back(std::make_pair(*this, Derivation(values(this->get_buffer().shape(), 1).tensor_cast(this->get_buffer().type(), false), derive_reduce_sum)));
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf));
		}
    }
}
