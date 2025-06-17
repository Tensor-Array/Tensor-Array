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

#define USING_DATA_TYPE_NVIDIA_FLOAT_8 (__nv_fp8_e5m2)(__nv_fp8_e4m3)
#define USING_DATA_TYPE_NVIDIA_FLOAT (__half)(__nv_bfloat16)
#define USING_DATA_TYPE_FLOAT (float)(double)
#define USING_DATA_TYPE_SINT (int8_t)(int16_t)(int32_t)(int64_t)
#define USING_DATA_TYPE_UINT (uint8_t)(uint16_t)(uint32_t)(uint64_t)
#define USING_DATA_TYPE USING_DATA_TYPE_SINT USING_DATA_TYPE_UINT USING_DATA_TYPE_FLOAT USING_DATA_TYPE_NVIDIA_FLOAT

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
		__global__ void set_values(T value_arr[], T value, unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_arr[thread_x] = value;
		}

		__global__ void set_values_random(float value_arr[], unsigned long long seed, unsigned int max_size)
		{
			curandState thisState;
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			curand_init(seed, thread_x, CURAND_RNG_TEST, &thisState);
			if (thread_x < max_size)
				value_arr[thread_x] = curand_uniform(&thisState);
		}

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

		__global__ void exp_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = expf(value_in[thread_x]);
		}

		__global__ void exp_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = exp(value_in[thread_x]);
		}

		__global__ void ln_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = logf(value_in[thread_x]);
		}

		__global__ void ln_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = log(value_in[thread_x]);
		}

		__global__ void sin_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = sinf(value_in[thread_x]);
		}

		__global__ void sin_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = sin(value_in[thread_x]);
		}

		__global__ void cos_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = cosf(value_in[thread_x]);
		}

		__global__ void cos_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = cos(value_in[thread_x]);
		}

		__global__ void tan_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = tanf(value_in[thread_x]);
		}

		__global__ void tan_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = tan(value_in[thread_x]);
		}

		__global__ void sinh_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = sinhf(value_in[thread_x]);
		}

		__global__ void sinh_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = sinh(value_in[thread_x]);
		}

		__global__ void cosh_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = coshf(value_in[thread_x]);
		}

		__global__ void cosh_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = cosh(value_in[thread_x]);
		}

		__global__ void tanh_arr(float value_out[], const float value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = tanhf(value_in[thread_x]);
		}

		__global__ void tanh_arr(double value_out[], const double value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = tanh(value_in[thread_x]);
		}

		template <typename T>
		__global__ void sigmoid_arr(T value_out[], const T value_in[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				value_out[thread_x] = T(1) / (T(1) + T(exp(double(-value_in[thread_x]))));
		}

		template <typename T>
		__global__ void pow_arr(T value_out[], const T a[], const T b[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				if constexpr (sizeof(T) > 32)
					value_out[thread_x] = pow(a[thread_x], b[thread_x]);
				else
					value_out[thread_x] = powf(a[thread_x], b[thread_x]);
		}

		template <typename T>
		__global__ void arr_more_than(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] > in2_value[thread_x];
		}

		template <typename T>
		__global__ void arr_less_than(bool out_value[], const T in1_value[], const T in2_value[], unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = in1_value[thread_x] < in2_value[thread_x];
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
		__global__ void array_condition(T out_value[], unsigned int c_size, const bool bool_value[], const T true_value[], const T false_value[])
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				out_value[thread_x] = bool_value[thread_x] ? true_value[thread_x] : false_value[thread_x];
		}

		__global__ void kernel_transpose(void* output, const void* input, unsigned int c_size, unsigned int dim_1_size, unsigned int dim_2_size , size_t child_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
			unsigned int thread_z = blockIdx.z * blockDim.z + threadIdx.z;
			if (thread_x < c_size && thread_y < dim_1_size && thread_z < dim_2_size)
				std::memcpy
				(
					reinterpret_cast<void*>(reinterpret_cast<size_t>(output) +
						thread_x * dim_2_size * dim_1_size * child_size +
						thread_z * dim_1_size * child_size +
						thread_y * child_size),
					reinterpret_cast<const void*>(reinterpret_cast<size_t>(input) +
						thread_x * dim_1_size * dim_2_size * child_size +
						thread_y * dim_2_size * child_size +
						thread_z * child_size),
					child_size
				);
		}

		bool equal_dim_size(const TensorBase& a, const TensorBase& b);

		Tensor derive_transpose(const Tensor& a, const Tensor&, bool is_derive, const DataBuffer&)
		{
			const std::vector<unsigned int> shape_a = a.get_buffer().shape();
			assert(shape_a.size() == 4);
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(Tensor(), derive_transpose)));
			}
			cudaError cudaStat;
			devices::Device this_cuda{ devices::CUDA };
			cudaStat = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cudaStat = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			void* c_ptr;
			cudaStat = cudaMalloc(&c_ptr, base_a.data_size());
			cudaStat = cudaMemset(c_ptr, 0, a.get_buffer().data_size());
			dim3 block_dim(8, 16, 8);
			dim3 grid_dim(
				shape_a.begin()[0] / block_dim.x + (shape_a.begin()[0] % block_dim.x ? 1U : 0U),
				shape_a.begin()[1] / block_dim.y + (shape_a.begin()[1] % block_dim.y ? 1U : 0U),
				shape_a.begin()[2] / block_dim.z + (shape_a.begin()[2] % block_dim.z ? 1U : 0U)
			);
			kernel_transpose<<<grid_dim, block_dim>>>(c_ptr, base_a.data(), shape_a.begin()[0], shape_a.begin()[1], shape_a.begin()[2], shape_a.begin()[3] * get_sizeof_type(a.get_buffer().type()));
			cudaStat = cudaDeviceSynchronize();
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cudaStat));
			}
			TensorBase value_buf(a.get_buffer().type(), { shape_a.begin()[0], shape_a.begin()[2], shape_a.begin()[1], shape_a.end()[-1]}, c_ptr, this_cuda);
			cudaStat = cudaFree(c_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
		}
		
		template<typename T>
		Tensor values0(const std::initializer_list<unsigned int>& list_dim, T value)
		{
			cudaError_t cudaStatus;
			T* dev_ptr;
			unsigned int total_size = 1;
			for (unsigned int i: list_dim)
				total_size *= i;
			devices::Device this_cuda{ devices::CUDA };
			cudaStatus = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cudaStatus = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			cudaStatus = cudaMalloc(&dev_ptr, total_size * sizeof(float));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(total_size / block_dim.x + (total_size % block_dim.x ? 1U : 0U));
			set_values << <grid_dim, block_dim>> > (dev_ptr, value, total_size);
			cudaStatus = cudaDeviceSynchronize();
			assert((cudaStatus = cudaGetLastError()) == cudaSuccess);
			if (cudaStatus != cudaSuccess);
			TensorBase other_buf(typeid(T), list_dim, dev_ptr, this_cuda);
			cudaStatus = cudaFree(dev_ptr);
			return Tensor(std::move(other_buf));
		}

#define ADD_CODE(TYPE) \
Tensor values(const std::initializer_list<unsigned int>& list_dim, TYPE value) \
{ \
return values0<TYPE>(list_dim, value); \
}
		LOOP(USING_DATA_TYPE);
#undef ADD_CODE

		Tensor tensor_rand(const std::initializer_list<unsigned int>& list_dim, unsigned int seed)
		{
			cudaError_t cudaStatus;
			float* dev_ptr;
			unsigned int total_size = 1;
			for (unsigned int i: list_dim)
				total_size *= i;
			devices::Device this_cuda{ devices::CUDA };
			cudaStatus = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cudaStatus = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			cudaStatus = cudaMalloc(&dev_ptr, total_size * sizeof(float));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(total_size / block_dim.x + (total_size % block_dim.x ? 1U : 0U));
			set_values_random << <grid_dim, block_dim>> > (dev_ptr, seed, total_size);
			cudaStatus = cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cudaStatus));
			}
			TensorBase other_buf(typeid(float), list_dim, dev_ptr, this_cuda);
			cudaStatus = cudaFree(dev_ptr);
			return other_buf;
		}

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
arr_more_than<<<grid_dim, block_dim>>>(c_ptr, static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size);
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

		Tensor condition(const Tensor& bool_value, const Tensor& true_value, const Tensor& false_value, bool is_derive)
		{
			assert(
				equal_dim_size(bool_value.get_buffer(), true_value.get_buffer()) &&
				equal_dim_size(bool_value.get_buffer(), false_value.get_buffer()) &&
				bool_value.get_buffer().type() == typeid(bool) &&
				true_value.get_buffer().type() == false_value.get_buffer().type()
			);
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				Tensor value_ones = values(bool_value.get_buffer().shape(), 1).tensor_cast(true_value.get_buffer().type(), false);
				Tensor value_zeros = zeros<uint32_t>(bool_value.get_buffer().shape()).tensor_cast(false_value.get_buffer().type(), false);
				temp.push_back(std::make_pair(true_value, Derivation(condition(bool_value, value_ones, value_zeros, false), multiply)));
				temp.push_back(std::make_pair(false_value, Derivation(condition(bool_value, value_zeros, value_ones, false), multiply)));
			}
			cudaError cuda_status;
			TensorBase other_buf;
			void* ptr_out;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_bool_value = bool_value.get_buffer().change_device(this_cuda);
			TensorBase base_true_value = true_value.get_buffer().change_device(this_cuda);
			TensorBase base_false_value = false_value.get_buffer().change_device(this_cuda);
			std::size_t c_size = base_bool_value.data_size();
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(c_size / block_dim.x + (c_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(true_value.get_buffer().type() == typeid(TYPE) && false_value.get_buffer().type() == typeid(TYPE)) \
{ \
cuda_status = cudaMalloc(&ptr_out, c_size * sizeof(TYPE));\
array_condition<<<grid_dim, block_dim>>>(static_cast<TYPE*>(ptr_out), c_size, static_cast<const bool*>(base_bool_value.data()), static_cast<const TYPE*>(base_true_value.data()), static_cast<const TYPE*>(base_false_value.data())); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), bool_value.get_buffer().shape(), ptr_out, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(ptr_out);
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

		Tensor power(const Tensor& a, const Tensor& b, bool is_derive)
		{
			assert(equal_dim_size(a.get_buffer(), b.get_buffer()));
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(multiply(b, power(a, add(b, values(b.get_buffer().shape(), -1).tensor_cast(b.get_buffer().type(), false), false), false), false, nullptr), multiply)));
				temp.push_back(std::make_pair(b, Derivation(multiply(a.log(), power(a, b, false), false, nullptr), multiply)));
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
pow_arr<<<block_dim, grid_dim>>>(static_cast<TYPE*>(c_ptr), static_cast<const TYPE*>(base_a.data()), static_cast<const TYPE*>(base_b.data()), c_size); \
cuda_status = cudaDeviceSynchronize(); \
other_buf = TensorBase(typeid(TYPE), a.get_buffer().shape(), c_ptr, this_cuda); \
}
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaFree(c_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::exp(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(*this, Derivation(this->exp(false), multiply)));
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
exp_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::log() const
		{
			std::vector<std::pair<Tensor, Derivation>> temp =
			{
				std::make_pair(*this, Derivation(divide(values(this->get_buffer().shape(), 1).tensor_cast(this->get_buffer().type(), false), *this, false), multiply)),
			};
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
ln_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::sin(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(*this, Derivation(this->cos(false), multiply)));
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
sin_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::cos(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(*this, Derivation(-this->sin(false), multiply)));
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
cos_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::tan(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				Tensor temp_ones = values(this->get_buffer().shape(), 1.f).tensor_cast(this->get_buffer().type(), false);
				Tensor temp_twos = values(this->get_buffer().shape(), 2.f).tensor_cast(this->get_buffer().type(), false);
				temp.push_back(std::make_pair(*this, Derivation(divide(temp_ones, divide(add(temp_ones, multiply(temp_twos, *this, false, DataBuffer()).cos(false)), temp_twos, false), false), multiply)));
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
tan_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::sinh(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(*this, Derivation(this->cosh(false), multiply)));
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
sinh_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::cosh(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(*this, Derivation(this->sinh(false), multiply)));
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
cosh_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::tanh(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				Tensor temp_ones = values(this->get_buffer().shape(), 1.f).tensor_cast(this->get_buffer().type(), false);
				Tensor temp_twos = values(this->get_buffer().shape(), 2.f).tensor_cast(this->get_buffer().type(), false);
				temp.push_back(std::make_pair(*this, Derivation(divide(temp_ones, divide(add(temp_ones, multiply(temp_twos, *this, false, DataBuffer()).cosh(false)), temp_twos, false), false), multiply)));
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
tanh_arr<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_of_this.data()), out_size);
			LOOP((float)(double));
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			TensorBase other_buf(this->get_buffer().type(), this->get_buffer().shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
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
