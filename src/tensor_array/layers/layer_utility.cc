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

#include "layer_utility.hh"

namespace tensor_array
{
    namespace layers
    {
        ActivationImpl::ActivationImpl(LayerFunction func) :
            func(func)
        {
        }

        value::Tensor ActivationImpl::calculate(const value::Tensor& input)
        {
            return this->func(input);
        }

        ReShapeImpl::ReShapeImpl(const std::initializer_list<unsigned int>& shape) :
            shape(shape)
        {
        }

        value::Tensor ReShapeImpl::calculate(const value::Tensor& input)
        {
            std::vector<unsigned int> shape = this->shape;
            shape.insert(shape.begin(), input.get_buffer().shape().begin()[0]);
            return input.reshape(shape);
        }

        value::Tensor NoActivation(const value::Tensor& input)
        {
            return input;
        }

        value::Tensor Sigmoid(const value::Tensor& input)
        {
            return input.sigmoid();
        }
        value::Tensor ReLU(const value::Tensor& input)
        {
            value::Tensor temp_zeros = value::zeros<float>(input.get_buffer().shape());
            return condition(input > temp_zeros, input, temp_zeros);
        }

        value::Tensor tanh(const value::Tensor& input)
        {
            return input.tanh();
        }

        value::Tensor SoftMax(const value::Tensor& input, unsigned char dim)
        {
            if (dim)
            {
                std::vector<value::Tensor> temp;
                for (value::Tensor ten : input)
                    temp.push_back(SoftMax(ten, dim - 1));
                return add_dim(temp);
            }
            value::Tensor a1 = input.exp();
            value::Tensor a2 = a1.value_scalar();
            return a1 / a2;
        }
    }
}