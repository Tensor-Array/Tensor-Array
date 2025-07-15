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

#include "layer_holder.hh"
#pragma once

namespace tensor_array
{
	namespace layers
	{
        class CUDA_ML_API LinearImpl final :
            public TensorCalculateLayerImpl
        {
        private:
            value::Tensor weight;
            value::Tensor bias;
        public:
            void layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&&) override;
            value::Tensor calculate(const value::Tensor&) override;
            LinearImpl(unsigned int);
        };

        using Linear = LayerHolder<LinearImpl>;

        inline void test()
        {
            Linear test1(1);
            test1(tensor_array::value::Tensor());
        }
	}
}


