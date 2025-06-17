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
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif // !TENSOR_CONTENT

#define LOOP(seq) END(A seq)
#define BODY(x) ADD_CODE(x)
#define A(x) BODY(x) B
#define B(x) BODY(x) A
#define A_END
#define B_END
#define END(...) END_(__VA_ARGS__)
#define END_(...) __VA_ARGS__##_END

#define USING_DATA_TYPE_NVIDIA_FLOAT_8 (__nv_fp8_e4m3)(__nv_fp8_e5m2)
#define USING_DATA_TYPE_NVIDIA_FLOAT (__half)(__nv_bfloat16)
#define USING_DATA_TYPE_FLOAT (float)(double)
#define USING_DATA_TYPE_SINT (int16_t)(int32_t)(int64_t)
#define USING_DATA_TYPE_UINT (uint16_t)(uint32_t)(uint64_t)

#define USING_DATA_TYPE_CAST_FROM \
(__nv_fp8_e4m3) \
USING_DATA_TYPE_SINT \
USING_DATA_TYPE_UINT \
USING_DATA_TYPE_FLOAT \
USING_DATA_TYPE_NVIDIA_FLOAT

#define USING_DATA_TYPE_CAST_TO \
(bool) \
(int8_t) \
(uint8_t) \
USING_DATA_TYPE_CAST_FROM

namespace tensor_array
{
    namespace value
    {
        template <typename T_O, typename T_I>
		__global__ void type_casting(T_O* output, const T_I* input, unsigned int c_size)
		{
			unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_x < c_size)
				output[thread_x] = static_cast<T_O>(input[thread_x]);
		}
        
        Tensor derive_reshape_cast(const Tensor& dat, const Tensor& new_shape, bool, const DataBuffer&);

        template <typename T>
		Tensor Tensor::cast(bool is_derive) const
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
				temp.push_back(std::make_pair(*this, Derivation(*this, derive_reshape_cast)));
			cudaError cuda_status;
			T* out_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cuda_status = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cuda_status = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_of_this = this->get_buffer().change_device(this_cuda);
			std::size_t total_size = base_of_this.data_size() / get_sizeof_type(base_of_this.type());
			cuda_status = cudaMalloc(&out_ptr, total_size * sizeof(T));
			dim3 block_dim(cu_dev_prop.maxThreadsDim[0]);
			dim3 grid_dim(total_size / block_dim.x + (total_size % block_dim.x ? 1U : 0U));
#define ADD_CODE(TYPE) \
if(this->get_buffer().type() == typeid(TYPE)) \
type_casting<<<grid_dim, block_dim>>>(out_ptr, static_cast<const TYPE*>(base_of_this.data()), total_size);
			LOOP(USING_DATA_TYPE_CAST_FROM);
#undef ADD_CODE
			cuda_status = cudaDeviceSynchronize();
			cuda_status = cudaGetLastError();
			if (cuda_status != cudaSuccess)
			{
				std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_status));
			}
			std::type_index test = typeid(T);
			if (dynamic_type_size.find(test) == dynamic_type_size.end())
				dynamic_type_size.insert(std::make_pair(test, sizeof(T)));
			TensorBase other_buf(typeid(T), base_of_this.shape(), out_ptr, this_cuda);
			cuda_status = cudaFree(out_ptr);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor Tensor::tensor_cast(const std::type_info& dtype, bool is_derive) const
		{
			if (this->get_buffer().type() == dtype)
				return *this;
#define ADD_CODE(TYPE) \
if(dtype == typeid(TYPE)) \
return this->cast<TYPE>(is_derive);
			LOOP(USING_DATA_TYPE_CAST_TO);
#undef ADD_CODE
			throw std::exception();
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
