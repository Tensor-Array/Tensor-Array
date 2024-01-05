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

#include "attention.hh"
#pragma once

namespace tensor_array
{
    namespace layers
    {
        class CUDA_ML_API TransformerEncoderImpl final :
            public TensorCalculateLayerImpl
        {
        private:
            MultiHeadAttention multihead_attn;
            Sequential feed_forward;
            Normalization
                layer_norm_1 = Normalization(std::initializer_list<unsigned char>{1}),
                layer_norm_2 = Normalization(std::initializer_list<unsigned char>{1});
        public:
            TransformerEncoderImpl(unsigned int, unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&) override;
        };

        using TransformerEncoder = LayerHolder<TransformerEncoderImpl>;

        class CUDA_ML_API TransformerDecoderImpl final :
            public LayerImpl
        {
        private:
            MultiHeadAttention masked_multihead_attn, multihead_attn;
            Sequential feed_forward;
            Normalization
                layer_norm_1 = Normalization(std::initializer_list<unsigned char>{1}),
                layer_norm_2 = Normalization(std::initializer_list<unsigned char>{1}),
                layer_norm_3 = Normalization(std::initializer_list<unsigned char>{1});
        public:
            TransformerDecoderImpl(unsigned int, unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&);
        };

        using TransformerDecoder = LayerHolder<TransformerDecoderImpl>;

        class TransformerImpl final :
            public LayerImpl
        {
        private:
            Sequential encoder_blocks;
            std::vector<TransformerDecoder> decoder_blocks;
            Linear fc;
        public:
            TransformerImpl(unsigned int, unsigned int, unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&);
        };

        using Transformer = LayerHolder<TransformerImpl>;
    }
}

