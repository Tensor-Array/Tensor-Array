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
#include <cstdio>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif

#if __CUDA_ARCH__ >= 800
#define USE_BF16 (__nv_bfloat16)
#else
#define USE_BF16
#endif

#if __CUDA_ARCH__ >= 700
#define USE_FP16 (__half)
#else
#define USE_FP16
#endif

#if __CUDA_ARCH__ >= 600
#define USE_FP64 (double)
#else
#define USE_FP64
#endif

#define USING_DATA_TYPE_NVIDIA_FLOAT_8 (__nv_fp8_e5m2)(__nv_fp8_e4m3)
#define USING_DATA_TYPE_NVIDIA_FLOAT USE_FP16 USE_BF16
#define USING_DATA_TYPE_FLOAT (float)USE_FP64
#define USING_DATA_TYPE_SINT (int32_t)
#define USING_DATA_TYPE_UINT (uint32_t)(unsigned long long int)
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
		__host__ __device__ dimension operator+(const dimension& a, const dimension& b)
		{
			dimension temp;
			temp.x = a.x + b.x;
			temp.y = a.y + b.y;
			temp.z = a.z + b.z;
			return temp;
		}

		__host__ __device__ dimension operator-(const dimension& a, const dimension& b)
		{
			dimension temp;
			temp.x = a.x - b.x;
			temp.y = a.y - b.y;
			temp.z = a.z - b.z;
			return temp;
		}

		__host__ __device__ dimension operator*(const dimension& a, const dimension& b)
		{
			dimension temp;
			temp.x = a.x * b.x;
			temp.y = a.y * b.y;
			temp.z = a.z * b.z;
			return temp;
		}

		__host__ __device__ dimension operator/(const dimension& a, const dimension& b)
		{
			dimension temp;
			temp.x = a.x / b.x;
			temp.y = a.y / b.y;
			temp.z = a.z / b.z;
			return temp;
		}

		template <typename T>
		__global__ void kernel_derive_conv_padding
		(
			T* dst,
			const T* src,
			unsigned int batch_size,
			unsigned int channel_count,
			dimension input_dim,
			dimension padding
		)
		{
			dimension thread_value;
			thread_value.x = blockIdx.x * blockDim.x + threadIdx.x;
			thread_value.y = blockIdx.y * blockDim.y + threadIdx.y;
			thread_value.z = blockIdx.z * blockDim.z + threadIdx.z;
			unsigned int thread_value_multi = thread_value.x * thread_value.y * thread_value.z;
			for (unsigned int i = 0; i < batch_size; i++)
				for (unsigned int j = 0; j < channel_count; j++)
					dst[i * channel_count * thread_value_multi + j * thread_value_multi + thread_value.x * input_dim.y * input_dim.z + thread_value.y * input_dim.z + thread_value.z] =
					src[i * channel_count * thread_value_multi + j * thread_value_multi + (thread_value.x + padding.x) * input_dim.y * input_dim.z + (thread_value.y + padding.y) * input_dim.z + (thread_value.z + padding.z)];
		}

		Tensor convolution_padding(const Tensor& a, const Tensor&, bool is_derive, const DataBuffer& data_buf);

		Tensor derive_convolution_padding(const Tensor& a, const Tensor&, bool is_derive, const DataBuffer& data_buf)
		{
			std::initializer_list<unsigned int> a_shape = a.get_buffer().shape();
			assert(data_buf.get_data_size() == sizeof(ConvolutionParameter));
			const dimension& param = *static_cast<const dimension*>(data_buf.get_data());
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(a, Derivation(Tensor(), convolution_padding, false, param)));
			cudaError cuda_status;
			void* out_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			dimension input_size =
			{
				a.get_buffer().shape().begin() + 2 < a.get_buffer().shape().end() ? a.get_buffer().shape().begin()[2] - param.x * 2 : 1U,
				a.get_buffer().shape().begin() + 3 < a.get_buffer().shape().end() ? a.get_buffer().shape().begin()[3] - param.y * 2 : 1U,
				a.get_buffer().shape().begin() + 4 < a.get_buffer().shape().end() ? a.get_buffer().shape().begin()[4] - param.z * 2 : 1U,
			};

			std::vector<unsigned int> new_shape;
			new_shape.push_back(a.get_buffer().shape().begin()[0]);
			new_shape.push_back(a.get_buffer().shape().begin()[1]);
			if (a.get_buffer().shape().begin() + 2 < a.get_buffer().shape().end())
				new_shape.push_back(input_size.x);
			if (a.get_buffer().shape().begin() + 3 < a.get_buffer().shape().end())
				new_shape.push_back(input_size.y);
			if (a.get_buffer().shape().begin() + 4 < a.get_buffer().shape().end())
				new_shape.push_back(input_size.z);
			std::size_t output_data_size = 1;
			for (auto& it : new_shape)
				output_data_size *= it;

			cuda_status = cudaMalloc(&out_ptr, output_data_size * get_sizeof_type(a.get_buffer().type()));
			dim3 block_dim(8, 16, 8);
			dim3 grid_dim
			(
				input_size.x / block_dim.x + (input_size.x % block_dim.x ? 1U : 0U),
				input_size.y / block_dim.y + (input_size.y % block_dim.y ? 1U : 0U),
				input_size.z / block_dim.z + (input_size.z % block_dim.z ? 1U : 0U)
			);

#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE)) \
kernel_derive_conv_padding<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_a.data()), a_shape.begin()[0], a_shape.begin()[1], input_size, param);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			cuda_status = cudaGetLastError();
			if (cuda_status != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_status));
			}
			TensorBase value_buf(a.get_buffer().type(), new_shape, out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
		}

		template <typename T>
		__global__ void kernel_conv_padding
		(
			T* dst,
			const T* src,
			unsigned int batch_size,
			unsigned int channel_count,
			dimension input_dim,
			dimension padding
		)
		{
			dimension thread_value;
			thread_value.x = blockIdx.x * blockDim.x + threadIdx.x;
			thread_value.y = blockIdx.y * blockDim.y + threadIdx.y;
			thread_value.z = blockIdx.z * blockDim.z + threadIdx.z;
			unsigned int thread_value_multi = thread_value.x * thread_value.y * thread_value.z;
			for (unsigned int i = 0; i < batch_size; i++)
				for (unsigned int j = 0; j < channel_count; j++)
					dst[i * channel_count * thread_value_multi + j * thread_value_multi + (thread_value.x + padding.x) * input_dim.y * input_dim.z + (thread_value.y + padding.y) * input_dim.z + (thread_value.z + padding.z)] =
					src[i * channel_count * thread_value_multi + j * thread_value_multi + thread_value.x * input_dim.y * input_dim.z + thread_value.y * input_dim.z + thread_value.z];
		}


		Tensor convolution_padding(const Tensor& a, const Tensor&, bool is_derive, const DataBuffer& data_buf)
		{
			std::initializer_list<unsigned int> a_shape = a.get_buffer().shape();
			assert(data_buf.get_data_size() == sizeof(ConvolutionParameter));
			const dimension& param = *static_cast<const dimension*>(data_buf.get_data());
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(a, Derivation(Tensor(), derive_convolution_padding, false, param)));
			cudaError cuda_status;
			void* out_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			dimension input_size =
			{
				a.get_buffer().shape().begin() + 2 < a.get_buffer().shape().end() ? a.get_buffer().shape().begin()[2] : 1U,
				a.get_buffer().shape().begin() + 3 < a.get_buffer().shape().end() ? a.get_buffer().shape().begin()[3] : 1U,
				a.get_buffer().shape().begin() + 4 < a.get_buffer().shape().end() ? a.get_buffer().shape().begin()[4] : 1U,
			};

			std::vector<unsigned int> new_shape;
			new_shape.push_back(a.get_buffer().shape().begin()[0]);
			new_shape.push_back(a.get_buffer().shape().begin()[1]);
			if (a.get_buffer().shape().begin() + 2 < a.get_buffer().shape().end())
				new_shape.push_back(input_size.x + param.x * 2);
			if (a.get_buffer().shape().begin() + 3 < a.get_buffer().shape().end())
				new_shape.push_back(input_size.y + param.y * 2);
			if (a.get_buffer().shape().begin() + 4 < a.get_buffer().shape().end())
				new_shape.push_back(input_size.z + param.z * 2);

			std::size_t output_data_size = 1;
			for (auto& it : new_shape)
				output_data_size *= it;

			cuda_status = cudaMalloc(&out_ptr, output_data_size * get_sizeof_type(a.get_buffer().type()));
			dim3 block_dim(8, 16, 8);
			dim3 grid_dim
			(
				input_size.x / block_dim.x + (input_size.x % block_dim.x ? 1U : 0U),
				input_size.y / block_dim.y + (input_size.y % block_dim.y ? 1U : 0U),
				input_size.z / block_dim.z + (input_size.z % block_dim.z ? 1U : 0U)
			);

#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE)) \
kernel_conv_padding<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_a.data()), a_shape.begin()[0], a_shape.begin()[1], input_size, param);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			cuda_status = cudaGetLastError();
			if (cuda_status != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_status));
			}
			TensorBase value_buf(a.get_buffer().type(), new_shape, out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
		}

		template <typename T>
		__global__ void kernel_col2im_2
		(
			T* dst,
			const T* src,
			ConvolutionParameter param,
			dimension input_start
		)
		{
			dimension thread_value;
			thread_value.x;
			thread_value.y;
			thread_value.z;
			if (thread_value.x < param.kernel.x && thread_value.y < param.kernel.y && thread_value.z < param.kernel.z)
				atomicAdd
				(
					dst +
					(input_start.x * param.strides.x + thread_value.x * param.dilation.x) * param.input.y * param.input.z +
					(input_start.y * param.strides.y + thread_value.y * param.dilation.y) * param.input.z +
					(input_start.z * param.strides.z + thread_value.z * param.dilation.z),
					src[thread_value.x * param.kernel.y * param.kernel.z + thread_value.y * param.kernel.z + thread_value.z]
				);
		}

		template <typename T>
		__global__ void kernel_col2im
		(
			T* dst,
			const T* src,
			unsigned int batch_size,
			unsigned int channel_count,
			ConvolutionParameter param
		)
		{
			dimension thread_value;
			thread_value.x = blockIdx.x * blockDim.x + threadIdx.x;
			thread_value.y = blockIdx.y * blockDim.y + threadIdx.y;
			thread_value.z = blockIdx.z * blockDim.z + threadIdx.z;
			dimension output_size = ((param.input - param.dilation * (param.kernel - dimension()) - dimension()) / param.strides) + dimension();
			unsigned int input_xyz = param.input.x * param.input.y * param.input.z;
			unsigned int filter_xyz = param.kernel.x * param.kernel.y * param.kernel.z;
			unsigned int back_channel_filter = batch_size * channel_count * filter_xyz;
			for (unsigned int i = 0; i < batch_size; i++)
				for (unsigned int j = 0; j < channel_count; j++)
					if (thread_value.x < output_size.x && thread_value.y < output_size.y && thread_value.z < output_size.z)
					{
						dim3 thread_per_block = blockDim;
						dim3 block_per_grid1
						(
							param.kernel.x / thread_per_block.x + (param.kernel.x % thread_per_block.x ? 1U : 0U),
							param.kernel.y / thread_per_block.y + (param.kernel.y % thread_per_block.y ? 1U : 0U),
							param.kernel.z / thread_per_block.z + (param.kernel.z % thread_per_block.z ? 1U : 0U)
						);
						kernel_col2im_2<T> << <block_per_grid1, thread_per_block >> >
							(
								dst +
								i * channel_count * input_xyz +
								j * input_xyz,
								src +
								thread_value.x * output_size.y * output_size.z * back_channel_filter +
								thread_value.y * output_size.z * back_channel_filter +
								thread_value.z * back_channel_filter +
								i * channel_count * filter_xyz +
								j * filter_xyz,
								param,
								thread_value
								);
					}
			__syncthreads();
		}

		Tensor convolution_im2col(const Tensor& a, const Tensor&, bool is_derive, const DataBuffer& data_buf);

		Tensor convolution_col2im(const Tensor& a, const Tensor&, bool is_derive, const DataBuffer& data_buf)
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			const ConvolutionParameter& param = *static_cast<const ConvolutionParameter*>(data_buf.get_data());
			if (is_derive)
				temp.push_back(std::make_pair(a, Derivation(Tensor(), convolution_im2col, false, param)));
			cudaError cuda_status;
			void* out_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			dimension output_size = ((param.input - param.dilation * (param.kernel - dimension()) - dimension()) / param.strides) + dimension();

			const std::vector<unsigned int> new_shape =
			{
				a.get_buffer().shape().begin()[1],
				a.get_buffer().shape().begin()[2],
				param.input.x,
				param.input.y,
				param.input.z
			};

			std::size_t output_data_size = 1;
			for (auto& it : new_shape)
				output_data_size *= it;

			cuda_status = cudaMalloc(&out_ptr, output_data_size * get_sizeof_type(a.get_buffer().type()));
			cuda_status = cudaMemset(out_ptr, 0, output_data_size * get_sizeof_type(a.get_buffer().type()));

			dim3 block_dim(8, 16, 8);
			dim3 grid_dim
			(
				output_size.x / block_dim.x + (output_size.x % block_dim.x ? 1U : 0U),
				output_size.y / block_dim.y + (output_size.y % block_dim.y ? 1U : 0U),
				output_size.z / block_dim.z + (output_size.z % block_dim.z ? 1U : 0U)
			);

#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE)) \
kernel_col2im<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_a.data()), a.get_buffer().shape().begin()[1], a.get_buffer().shape().begin()[2], param);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			cuda_status = cudaGetLastError();
			if (cuda_status != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_status));
			}
			TensorBase value_buf(a.get_buffer().type(), new_shape, out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
		}

		template <typename T>
		__global__ void kernel_im2col_2
		(
			T* dst,
			const T* src,
			ConvolutionParameter param,
			dimension input_start
		)
		{
			dimension thread_value;
			thread_value.x = blockIdx.x * blockDim.x + threadIdx.x;
			thread_value.y = blockIdx.y * blockDim.y + threadIdx.y;
			thread_value.z = blockIdx.z * blockDim.z + threadIdx.z;
			if (thread_value.x < param.kernel.x && thread_value.y < param.kernel.y && thread_value.z < param.kernel.z)
				dst[thread_value.x * param.kernel.y * param.kernel.z + thread_value.y * param.kernel.z + thread_value.z] =
				src
				[
					(input_start.x * param.strides.x + thread_value.x * param.dilation.x) * param.input.y * param.input.z +
						(input_start.y * param.strides.y + thread_value.y * param.dilation.y) * param.input.z +
						(input_start.z * param.strides.z + thread_value.z * param.dilation.z)
				];
		}

		template <typename T>
		__global__ void kernel_im2col
		(
			T* dst,
			const T* src,
			unsigned int batch_size,
			unsigned int channel_count,
			ConvolutionParameter param
		)
		{
			dimension thread_value;
			thread_value.x = blockIdx.x * blockDim.x + threadIdx.x;
			thread_value.y = blockIdx.y * blockDim.y + threadIdx.y;
			thread_value.z = blockIdx.z * blockDim.z + threadIdx.z;
			dimension output_size = ((param.input - param.dilation * (param.kernel - dimension()) - dimension()) / param.strides) + dimension();
			unsigned int input_xyz = param.input.x * param.input.y * param.input.z;
			unsigned int filter_xyz = param.kernel.x * param.kernel.y * param.kernel.z;
			unsigned int back_channel_filter = batch_size * channel_count * filter_xyz;
			for (unsigned int i = 0; i < batch_size; i++)
				for (unsigned int j = 0; j < channel_count; j++)
					if (thread_value.x < output_size.x && thread_value.y < output_size.y && thread_value.z < output_size.z)
					{
						dim3 thread_per_block = blockDim;
						dim3 block_per_grid1
						(
							param.kernel.x / thread_per_block.x + (param.kernel.x % thread_per_block.x ? 1U : 0U),
							param.kernel.y / thread_per_block.y + (param.kernel.y % thread_per_block.y ? 1U : 0U),
							param.kernel.z / thread_per_block.z + (param.kernel.z % thread_per_block.z ? 1U : 0U)
						);
						kernel_im2col_2<T> << <block_per_grid1, thread_per_block >> >
							(
								dst +
								thread_value.x * output_size.y * output_size.z * back_channel_filter +
								thread_value.y * output_size.z * back_channel_filter +
								thread_value.z * back_channel_filter +
								i * channel_count * filter_xyz +
								j * filter_xyz,
								src +
								i * channel_count * input_xyz +
								j * input_xyz,
								param,
								thread_value
								);
					}
			__syncthreads();
		}

		Tensor convolution_im2col(const Tensor& a, const Tensor&, bool is_derive, const DataBuffer& data_buf)
		{
			std::initializer_list<unsigned int> a_shape = a.get_buffer().shape();
			assert(data_buf.get_data_size() == sizeof(ConvolutionParameter));
			ConvolutionParameter param = *static_cast<const ConvolutionParameter*>(data_buf.get_data());
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(a, Derivation(Tensor(), convolution_col2im, false, param)));
			cudaError cuda_status;
			void* out_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			dimension output_size = ((param.input - param.dilation * (param.kernel - dimension()) - dimension()) / param.strides) + dimension();

			const std::vector<unsigned int> new_shape =
			{
				output_size.x * output_size.y * output_size.z,
				a_shape.begin()[0],
				a_shape.begin()[1],
				param.kernel.x * param.kernel.y * param.kernel.z
			};

			std::size_t output_data_size = 1;
			for (auto& it : new_shape)
				output_data_size *= it;

			cuda_status = cudaMalloc(&out_ptr, output_data_size * get_sizeof_type(a.get_buffer().type()));
			dim3 block_dim(8, 16, 8);
			dim3 grid_dim
			(
				output_size.x / block_dim.x + (output_size.x % block_dim.x ? 1U : 0U),
				output_size.y / block_dim.y + (output_size.y % block_dim.y ? 1U : 0U),
				output_size.z / block_dim.z + (output_size.z % block_dim.z ? 1U : 0U)
			);

#define ADD_CODE(TYPE) \
if(a.get_buffer().type() == typeid(TYPE)) \
kernel_im2col<<<grid_dim, block_dim>>>(static_cast<TYPE*>(out_ptr), static_cast<const TYPE*>(base_a.data()), a_shape.begin()[0], a_shape.begin()[1], param);
			LOOP(USING_DATA_TYPE);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			cuda_status = cudaGetLastError();
			if (cuda_status != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_status));
			}
			TensorBase value_buf(a.get_buffer().type(), new_shape, out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
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
