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

#include <cassert>
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif

namespace tensor_array
{
    namespace value
    {
        Tensor Tensor::convolution_convert(const ConvolutionParameter& param)
        {
            return convolution_im2col(*this, Tensor(), true, param);
        }

        Tensor Tensor::conv_padding(const dimension& pad) const
        {
            return convolution_padding(*this, Tensor(), true, pad);
        }

        Tensor convolution(const Tensor& input, const Tensor& kernel, const dimension& strides, const dimension& dilation)
        {
            std::initializer_list<unsigned int> input_shape = input.get_buffer().shape();
            std::initializer_list<unsigned int> filter_shape = kernel.get_buffer().shape();
            assert
            (
                input_shape.size() == filter_shape.size() &&
                input_shape.begin()[1] == filter_shape.begin()[0]
            );
            ConvolutionParameter param;
            param.input =
            {
                input_shape.begin() + 2 < input_shape.end() ? input_shape.begin()[2] : 1U,
                input_shape.begin() + 3 < input_shape.end() ? input_shape.begin()[3] : 1U,
                input_shape.begin() + 4 < input_shape.end() ? input_shape.begin()[4] : 1U
            };
            param.kernel =
            {
                filter_shape.begin() + 1 < filter_shape.end() - 1 ? filter_shape.begin()[1] : 1U,
                filter_shape.begin() + 2 < filter_shape.end() - 1 ? filter_shape.begin()[2] : 1U,
                filter_shape.begin() + 3 < filter_shape.end() - 1 ? filter_shape.begin()[3] : 1U
            };
            param.strides = strides;
            param.dilation = dilation;
            dimension output_size = ((param.input - param.dilation * (param.kernel - dimension()) - dimension()) / param.strides) + dimension();
            std::vector<unsigned int> final_shape;
            final_shape.push_back(input_shape.begin()[0]);
            final_shape.push_back(filter_shape.end()[-1]);
            if (input_shape.begin() + 2 < input_shape.end() && filter_shape.begin() + 1 < filter_shape.end() - 1)
                final_shape.push_back(output_size.x);
            if (input_shape.begin() + 3 < input_shape.end() && filter_shape.begin() + 2 < filter_shape.end() - 1)
                final_shape.push_back(output_size.y);
            if (input_shape.begin() + 4 < input_shape.end() && filter_shape.begin() + 3 < filter_shape.end() - 1)
                final_shape.push_back(output_size.z);
            DataBuffer dat_buf(param);
            Tensor encoded_input = convolution_im2col(input, Tensor(), true, dat_buf);
            assert
            (
                output_size.x * output_size.y * output_size.z == encoded_input.get_buffer().shape().begin()[0] &&
                param.kernel.x * param.kernel.y * param.kernel.z == encoded_input.get_buffer().shape().begin()[3]
            );
            unsigned int shape_m = encoded_input.get_buffer().shape().begin()[0] * encoded_input.get_buffer().shape().begin()[1];
            unsigned int shape_k = encoded_input.get_buffer().shape().begin()[2] * encoded_input.get_buffer().shape().begin()[3];
            Tensor temp = matmul(encoded_input.reshape({ shape_m, shape_k }),  kernel.reshape({ shape_k, filter_shape.end()[-1] }));
            temp = temp.reshape({ encoded_input.get_buffer().shape().begin()[0], input_shape.begin()[0] * filter_shape.end()[-1] });
            temp = temp.transpose(0, 1);
            temp = temp.reshape(final_shape);
            return temp;
        }
    }
}