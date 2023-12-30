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
            public LayerImpl,
            public CalculateStruct<value::Tensor, const value::Tensor&, const value::Tensor&, const value::Tensor&>
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
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&) override;
        };

        using TransformerDecoder = LayerHolder<TransformerDecoderImpl>;

        class TransformerImpl final :
            public LayerImpl,
            public CalculateStruct<value::Tensor, const value::Tensor&, const value::Tensor&, const value::Tensor&>
        {
        private:
            Sequential encoder_blocks;
            std::vector<TransformerDecoder> decoder_blocks;
            Linear fc;
        public:
            TransformerImpl(unsigned int, unsigned int, unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&) override;
        };

        using Transformer = LayerHolder<TransformerImpl>;
    }
}

