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

#include "linear.hh"

namespace tensor_array
{
	namespace layers
	{
        LinearImpl::LinearImpl(unsigned int units) :
            weight(),
            bias(value::values({ units }, 0.f))
        {
            this->map_tensor.insert(std::make_pair("bias", &bias));
            this->map_tensor.insert(std::make_pair("weight", &weight));
        }

        void LinearImpl::layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&& vector_shape)
        {
            auto& buf = vector_shape[0].first;
            if (!weight.has_tensor())
                weight = value::values({ buf.end()[-1], bias.get_buffer().shape().begin()[0] }, 1.f / buf.end()[-1]).tensor_cast(vector_shape[0].second).clone();
        }

        value::Tensor LinearImpl::calculate(const value::Tensor& input)
        {
            return matmul(input, weight) + bias;
        }
	}
}