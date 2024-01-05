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
#include "sequential.hh"
#include "normalization.hh"
#pragma once

namespace tensor_array
{
    namespace layers
    {
        value::Tensor CUDA_ML_API scaled_dot_product_attention(const value::Tensor&, const value::Tensor&, const value::Tensor&, const value::Tensor& = value::Tensor());

        class CUDA_ML_API MultiHeadAttentionImpl final :
            public LayerImpl
        {
        private:
            const unsigned int d_model, n_head;
            Linear w_q, w_k, w_v, w_o;
        public:
            MultiHeadAttentionImpl(unsigned int, unsigned int);
            void layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&&) override;
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&, const value::Tensor& = value::Tensor());
        };

        using MultiHeadAttention = LayerHolder<MultiHeadAttentionImpl>;
    }
}