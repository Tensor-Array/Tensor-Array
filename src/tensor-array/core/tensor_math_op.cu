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
#include <curand_kernel.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <exception>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif // !TENSOR_CONTENT

#define USING_DATA_TYPE_NVIDIA_FLOAT_8() (__nv_fp8_e5m2)(__nv_fp8_e4m3)
#define USING_DATA_TYPE_NVIDIA_FLOAT() (__half)(__nv_bfloat16)
#define USING_DATA_TYPE_FLOAT() (float)(double)
#define USING_DATA_TYPE_SINT() (int8_t)(int16_t)(int32_t)(int64_t)
#define USING_DATA_TYPE_UINT() (uint8_t)(uint16_t)(uint32_t)(uint64_t)
#if CUDART_VERSION >= 12020
#define USING_DATA_TYPE \
USING_DATA_TYPE_SINT() \
USING_DATA_TYPE_UINT() \
USING_DATA_TYPE_FLOAT() \
USING_DATA_TYPE_NVIDIA_FLOAT()
#else
#define USING_DATA_TYPE \
USING_DATA_TYPE_SINT() \
USING_DATA_TYPE_UINT() \
USING_DATA_TYPE_FLOAT()
#endif

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

		template <typename T>
		__global__ void sum_2_arr(T c[], const T a[], const T b[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				c[thread_x] = a[thread_x] + b[thread_x];
		}

		template <typename T>
		__global__ void mul_2_arr(T c[], const T a[], const T b[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				c[thread_x] = a[thread_x] * b[thread_x];
		}

		template <typename T>
		__global__ void div_2_arr(T c[], const T a[], const T b[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				c[thread_x] = a[thread_x] / b[thread_x];
		}

		template <typename T>
		__global__ void arr_greater_than(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] > in2_value[thread_x];
		}

        template <typename T>
		__global__ void arr_greater_equal(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] >= in2_value[thread_x];
		}

		template <typename T>
		__global__ void arr_less_than(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] < in2_value[thread_x];
		}

        template <typename T>
		__global__ void arr_less_equal(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] <= in2_value[thread_x];
		}

        template <typename T>
		__global__ void arr_equal_equal(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] == in2_value[thread_x];
		}

        template <typename T>
		__global__ void arr_not_equal(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] != in2_value[thread_x];
		}

		__global__ void arr_logical_and(bool out_value[], const bool in1_value[], const bool in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] && in2_value[thread_x];
		}


		__global__ void arr_logical_or(bool out_value[], const bool in1_value[], const bool in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] || in2_value[thread_x];
		}


		__global__ void arr_logical_not(bool out_value[], const bool in1_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = !in1_value[thread_x];
		}

        template <typename T>
		__global__ void sigmoid_arr(T value_out[], const T value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = T(1) / (T(1) + T(exp(double(-value_in[thread_x]))));
		}

		bool equal_dim_size(const TensorBase& a, const TensorBase& b);

		Tensor operator>(const Tensor& a, const Tensor& b)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			cudaError cuda_status;
			bool* c_ptr;
			Device this_cuda{ CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				base_a.data_size() / get_sizeof_type(base_a.type()),
				base_b.data_size() / get_sizeof_type(base_b.type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size * sizeof(bool));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
arr_greater_than<<<grid_dim, block_dim>>>(c_ptr, static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			cuda_status = cudaGetLastError();
			if (cuda_status != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_status));
			}
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

        Tensor operator>=(const Tensor& a, const Tensor& b)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			cudaError cuda_status;
			bool* c_ptr;
			Device this_cuda{ CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				base_a.data_size() / get_sizeof_type(base_a.type()),
				base_b.data_size() / get_sizeof_type(base_b.type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size * sizeof(bool));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
arr_greater_equal<<<grid_dim, block_dim>>>(c_ptr, static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			cuda_status = cudaGetLastError();
			if (cuda_status != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_status));
			}
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

		Tensor operator<(const Tensor& a, const Tensor& b)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			cudaError cuda_status;
			bool* c_ptr;
			Device this_cuda{CUDA};
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size * sizeof(bool));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
arr_less_than<<<grid_dim, block_dim>>>(c_ptr, static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

        Tensor operator<=(const Tensor& a, const Tensor& b)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			cudaError cuda_status;
			bool* c_ptr;
			Device this_cuda{CUDA};
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size * sizeof(bool));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
arr_less_equal<<<grid_dim, block_dim>>>(c_ptr, static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

        Tensor operator==(const Tensor& a, const Tensor& b)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			cudaError cuda_status;
			bool* c_ptr;
			Device this_cuda{CUDA};
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size * sizeof(bool));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
arr_equal_equal<<<grid_dim, block_dim>>>(c_ptr, static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

        Tensor operator!=(const Tensor& a, const Tensor& b)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			cudaError cuda_status;
			bool* c_ptr;
			Device this_cuda{CUDA};
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size * sizeof(bool));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
arr_not_equal<<<grid_dim, block_dim>>>(c_ptr, static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

		Tensor operator&&(const Tensor& a, const Tensor& b)
		{
			assert(
				equal_dim_size(a.get_buffer(), b.get_buffer())
				&& a.get_buffer().type() == typeid(bool)
				&& b.get_buffer().type() == typeid(bool)
			);
			cudaError cuda_status;
			bool* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size);
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
			arr_logical_and << <grid_dim, block_dim >> > (c_ptr, static_cast<const bool*>(base_a.data()), static_cast<const bool*>(base_b.data()), c_size);
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

		Tensor operator||(const Tensor& a, const Tensor& b)
		{
			assert(
				equal_dim_size(a.get_buffer(), b.get_buffer())
				&& a.get_buffer().type() == typeid(bool)
				&& b.get_buffer().type() == typeid(bool)
			);
			cudaError cuda_status;
			bool* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, c_size);
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
			arr_logical_or << <grid_dim, block_dim >> > (c_ptr, static_cast<const bool*>(base_a.data()), static_cast<const bool*>(base_b.data()), c_size);
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(typeid(bool), a.get_buffer().shape(), c_ptr, this_cuda);
			cuda_status = cudaFree(c_ptr);
			return other_buf;
		}

		Tensor Tensor::operator!()
		{
			assert(this->get_buffer().type() == typeid(bool));
			cudaError cuda_status;
			bool* out_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_of_this = this->get_buffer().change_device(this_cuda);
			cuda_status = cudaMalloc(&out_ptr, this->get_buffer().data_size());
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(this->get_buffer().data_size() / block_dim.x + 1U);
			arr_logical_not << < grid_dim, block_dim >> > (out_ptr, static_cast<const bool*>(base_of_this.data()), this->get_buffer().data_size());
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(typeid(bool), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return other_buf;
		}

		Tensor multiply(const Tensor& a, const Tensor& b, bool is_derive, const DataBuffer&)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(b.clone(), multiply)));
				temp.push_back(std::make_pair(b, Derivation(a.clone(), multiply)));
			}
			cudaError cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, std::max(a.get_buffer().data_size(), b.get_buffer().data_size()));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
{ \
mul_2_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), a.get_buffer().shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor add(const Tensor& a, const Tensor& b, bool is_derive)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(values(a.get_buffer().shape(), 1).tensor_cast(a.get_buffer().type(), false), multiply)));
				temp.push_back(std::make_pair(b, Derivation(values(b.get_buffer().shape(), 1).tensor_cast(b.get_buffer().type(), false), multiply)));
			}
			cudaError cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, std::max(a.get_buffer().data_size(), b.get_buffer().data_size()));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
{ \
sum_2_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), a.get_buffer().shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor divide(const Tensor& a, const Tensor& b, bool is_derive)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(divide(values(b.get_buffer().shape(), 1).tensor_cast(b.get_buffer().type(), false), b, false), multiply)));
				temp.push_back(std::make_pair(b, Derivation(divide(a, power(b, values(b.get_buffer().shape(), 2).tensor_cast(b.get_buffer().type(), false), false), false), multiply)));
			}
			cudaError cuda_status;
			TensorBase other_buf;
			void* c_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			std::size_t c_size = std::max
			(
				a.get_buffer().data_size() / get_sizeof_type(a.get_buffer().type()),
				b.get_buffer().data_size() / get_sizeof_type(b.get_buffer().type())
			);
			cuda_status = cudaMalloc(&c_ptr, std::max(a.get_buffer().data_size(), b.get_buffer().data_size()));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE) && b.get_buffer().type() == typeid(TYPE)) \
{ \
div_2_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), a.get_buffer().shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::sigmoid(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				Tensor temp_ones = values(this->get_buffer().shape(), 1.f).tensor_cast(this->get_buffer().type(), false);
				Tensor temp_sigmoid = this->sigmoid(false);
				temp.push_back(std::make_pair(*this, Derivation(multiply(temp_sigmoid, add(temp_ones, -temp_sigmoid, false), false, DataBuffer()), multiply)));
			}
			cudaError cuda_status;
			void* out_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_of_this = this->get_buffer().change_device(this_cuda);
			cuda_status = cudaMalloc(&out_ptr, this->get_buffer().data_size());
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			std::size_t out_size = this->get_buffer().data_size() / get_sizeof_type(this->get_buffer().type());
			dim3 grid_dim(out_size / block_dim.x + ((out_size % block_dim.x) ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(this->get_buffer().type() == typeid(TYPE)) \
sigmoid_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}
    }
}

#undef LOOP
#undef BODY
#undef A
#undef B
#undef A_END
#undef B_END
#undef END
#undef END_

#undef USING_DATA_TYPE
