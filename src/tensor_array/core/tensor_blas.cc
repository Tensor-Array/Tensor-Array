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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <complex>
#include <cassert>
#include <cstring>
#include <cuda_fp8.h>
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif // !TENSOR_CONTENT

namespace tensor_array
{
    namespace value
    {
        using namespace devices;
		bool equal_dim_size(const TensorBase& a, const TensorBase& b)
		{
			return std::vector<unsigned int>(a.shape()) == std::vector<unsigned int>(b.shape());
		}

		cudaDataType convert_cuda_type(const std::type_info& type)
		{
			if (type == typeid(float)) return CUDA_R_32F;
			if (type == typeid(double)) return CUDA_R_64F;
			if (type == typeid(__half)) return CUDA_R_16F;
			if (type == typeid(std::int8_t)) return CUDA_R_8I;
			if (type == typeid(std::complex<float>) || type == typeid(cuFloatComplex)) return CUDA_C_32F;
			if (type == typeid(std::complex<double>) || type == typeid(cuDoubleComplex)) return CUDA_C_64F;
			if (type == typeid(__half2)) return CUDA_C_16F;
			if (type == typeid(std::complex<std::int8_t>)) return CUDA_C_8I;
			if (type == typeid(std::uint8_t)) return CUDA_R_8U;
			if (type == typeid(std::complex<std::uint8_t>)) return CUDA_C_8U;
			if (type == typeid(std::int32_t)) return CUDA_R_32I;
			if (type == typeid(std::complex<std::int32_t>)) return CUDA_C_32I;
			if (type == typeid(std::uint32_t)) return CUDA_R_32U;
			if (type == typeid(std::complex<std::uint32_t>)) return CUDA_C_32U;
			if (type == typeid(__nv_bfloat16)) return CUDA_R_16BF;
			if (type == typeid(__nv_bfloat162)) return CUDA_C_16BF;
			if (type == typeid(std::int16_t)) return CUDA_R_16I;
			if (type == typeid(std::complex<std::int16_t>)) return CUDA_C_16I;
			if (type == typeid(std::uint16_t)) return CUDA_R_16U;
			if (type == typeid(std::complex<std::uint16_t>)) return CUDA_C_16U;
			if (type == typeid(std::int64_t)) return CUDA_R_64I;
			if (type == typeid(std::complex<std::int64_t>)) return CUDA_C_64I;
			if (type == typeid(std::uint64_t)) return CUDA_R_64U;
			if (type == typeid(__nv_fp8_e4m3)) return CUDA_R_8F_E4M3;
			if (type == typeid(__nv_fp8_e5m2)) return CUDA_R_8F_E5M2;
			throw std::exception();
		}

		Tensor dot(const Tensor& a, const Tensor& b, bool is_derive, const DataBuffer&)
		{
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(b, dot)));
				temp.push_back(std::make_pair(b, Derivation(a, dot)));
			}
			TensorBase other_buf;
			cudaError cudaStat;
			cublasHandle_t blasHandle;
			cublasStatus_t blasStat = cublasCreate(&blasHandle);
			blasStat = cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE);
			void* in_ptr;
			devices::Device this_cuda{ devices::CUDA };
			cudaStat = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cudaStat = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			if (equal_dim_size(base_a, base_b))
			{
				const std::type_info& c_type = comparison_type(base_a.type(), b.get_buffer().type());
				cudaStat = cudaMalloc(&in_ptr, get_sizeof_type(c_type));
				blasStat = cublasDotEx(blasHandle, base_a.data_size() / get_sizeof_type(base_a.type()),
					base_a.data(), convert_cuda_type(base_a.type()), 1,
					base_b.data(), convert_cuda_type(base_b.type()), 1,
					in_ptr, convert_cuda_type(c_type), (get_sizeof_type(c_type) < sizeof(float)) ? CUDA_R_32F : convert_cuda_type(c_type));
				other_buf = TensorBase(c_type, {}, in_ptr, this_cuda);
				cudaStat = cudaFree(in_ptr);
			}
			if (base_a.shape().size() == 0)
			{
				cudaStat = cudaMalloc(&in_ptr, base_b.data_size());
				cudaStat = cudaMemcpy(in_ptr, base_b.data(), base_b.data_size(), cudaMemcpyDeviceToDevice);
				blasStat = cublasScalEx(blasHandle, 0,
					base_a.data(), convert_cuda_type(base_a.type()),
					in_ptr, convert_cuda_type(base_b.type()),
					1, convert_cuda_type(base_a.type()));
				other_buf = TensorBase(base_b.type(), base_b.shape(), in_ptr, this_cuda);
				cudaStat = cudaFree(in_ptr);
			}
			if (base_b.shape().size() == 0)
			{
				cudaStat = cudaMalloc(&in_ptr, base_a.data_size());
				cudaStat = cudaMemcpy(in_ptr, base_a.data(), base_a.data_size(), cudaMemcpyDeviceToDevice);
				blasStat = cublasScalEx(blasHandle, 0,
					base_b.data(), convert_cuda_type(base_b.type()),
					in_ptr, convert_cuda_type(base_a.type()),
					1, convert_cuda_type(base_b.type()));
				other_buf = TensorBase(base_a.type(), base_a.shape(), in_ptr, this_cuda);
				cudaStat = cudaFree(in_ptr);
			}
			blasStat = cublasDestroy(blasHandle);
			return Tensor(std::move(other_buf), std::move(temp));
		}

		Tensor matmul(const Tensor& a, const Tensor& b, bool is_derive, const DataBuffer&)
		{
			cudaError cudaStat;
			devices::Device this_cuda{ devices::CUDA };
			cudaStat = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cudaStat = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			const std::initializer_list<unsigned int> shape_a = base_a.shape();
			const std::initializer_list<unsigned int> shape_b = base_b.shape();
			assert(shape_a.size() == 2 && shape_b.size() == 2 && shape_a.end()[-1] == shape_b.begin()[0]);
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(b.transpose(0, 1, false), matmul, false)));
				temp.push_back(std::make_pair(b, Derivation(a.transpose(0, 1, false), matmul, true)));
			}
			cublasHandle_t blasHandle;
			cublasStatus_t blasStat = cublasCreate(&blasHandle);
			blasStat = cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE);
			blasStat = cublasSetMathMode(blasHandle, CUBLAS_DEFAULT_MATH);
			void* c_ptr;
			const std::type_info& c_type = comparison_type(base_a.type(), base_b.type());
			cudaStat = cudaMalloc(&c_ptr, shape_a.begin()[0] * shape_b.end()[-1] * get_sizeof_type(c_type));
			cudaStat = cudaMemset(c_ptr, 0, shape_a.begin()[0] * shape_b.end()[-1] * get_sizeof_type(c_type));
			TensorBase alpha = values({}, 1).tensor_cast(c_type, false).get_buffer().change_device(this_cuda);
			TensorBase beta = zeros<uint32_t>({}).tensor_cast(c_type, false).get_buffer().change_device(this_cuda);
			blasStat = cublasGemmEx(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				shape_a.begin()[0], shape_b.end()[-1], shape_a.end()[-1],
				alpha.data(),
				base_a.data(), convert_cuda_type(base_a.type()), shape_a.begin()[0],
				base_b.data(), convert_cuda_type(base_b.type()), shape_b.begin()[0],
				beta.data(),
				c_ptr, convert_cuda_type(c_type), shape_a.begin()[0],
				convert_cuda_type(c_type), CUBLAS_GEMM_DEFAULT);
			blasStat = cublasDestroy(blasHandle);
			TensorBase value_buf(c_type, { shape_a.begin()[0] , shape_b.end()[-1] }, c_ptr, this_cuda);
			cudaStat = cudaFree(c_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
		}

		Tensor batchedmatmul(const Tensor& a, const Tensor& b, bool is_derive, const DataBuffer&)
		{
			cudaError cudaStat;
			devices::Device this_cuda{ devices::CUDA };
			cudaStat = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cudaStat = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			const std::initializer_list<unsigned int> shape_a = base_a.shape();
			const std::initializer_list<unsigned int> shape_b = base_b.shape();
			assert(shape_a.size() == shape_b.size() && std::memcmp(shape_a.begin(), shape_b.begin(), std::min(shape_a.size(), shape_b.size()) - 2) && shape_a.end()[-1] == shape_b.end()[-2]);
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(b.transpose(shape_b.size() - 2, shape_b.size() - 1, false), batchedmatmul, false)));
				temp.push_back(std::make_pair(b, Derivation(a.transpose(shape_a.size() - 2, shape_a.size() - 1, false), batchedmatmul, true)));
			}
			cublasHandle_t blasHandle;
			cublasStatus_t blasStat = cublasCreate(&blasHandle);
			blasStat = cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE);
			blasStat = cublasSetMathMode(blasHandle, CUBLAS_DEFAULT_MATH);
			void* c_ptr;
			const std::type_info& c_type = comparison_type(base_a.type(), base_b.type());

			unsigned int batch_size = 1;
			for (std::size_t i = 0; i < shape_a.size() - 2; i++)
				batch_size = shape_a.begin()[i];

			cudaStat = cudaMalloc(&c_ptr, batch_size * shape_a.end()[-2] * shape_b.end()[-1] * get_sizeof_type(c_type));
			cudaStat = cudaMemset(c_ptr, 0, batch_size * shape_a.end()[-2] * shape_b.end()[-1] * get_sizeof_type(c_type));
			TensorBase alpha = values({}, 1).tensor_cast(c_type, false).get_buffer().change_device(this_cuda);
			TensorBase beta = zeros<uint32_t>({}).tensor_cast(c_type, false).get_buffer().change_device(this_cuda);
			blasStat = cublasGemmStridedBatchedEx(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				shape_a.end()[-2], shape_b.end()[-1], shape_a.end()[-1],
				alpha.data(),
				base_a.data(), convert_cuda_type(base_a.type()), shape_a.end()[-2], 1,
				base_b.data(), convert_cuda_type(base_b.type()), shape_b.end()[-2], 1,
				beta.data(),
				c_ptr, convert_cuda_type(c_type), shape_a.end()[-2], 1, batch_size,
				convert_cuda_type(c_type), CUBLAS_GEMM_DEFAULT);
			blasStat = cublasDestroy(blasHandle);
            std::vector<unsigned int> out_dims = shape_a;
            out_dims[out_dims.size() - 1] = shape_b.end()[-1];

			TensorBase value_buf(c_type, out_dims, c_ptr, this_cuda);
			cudaStat = cudaFree(c_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
		}
    }
}