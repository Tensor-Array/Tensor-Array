#include "linear.hh"
#include "sequential.hh"
#include "normalization.hh"
#pragma once

namespace tensor_array
{
    namespace layers
    {
        class CUDA_ML_API MultiHeadAttentionImpl final :
            public LayerImpl,
            public CalculateStruct<value::Tensor, const value::Tensor&, const value::Tensor&, const value::Tensor&>
        {
        private:
            const unsigned int d_model, n_head;
            Linear w_q, w_k, w_v, w_o;
        public:
            MultiHeadAttentionImpl(unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&) override;
        };

        using MultiHeadAttention = LayerHolder<MultiHeadAttentionImpl>;

        class CUDA_ML_API TransformerEncoderImpl final : TensorCalculateLayerImpl 
        {
        private:
            MultiHeadAttention multihead_attn;
            Sequential feed_forward;
            Normalization
                layer_norm_1 = Normalization(std::initializer_list<unsigned char>{0}),
                layer_norm_2 = Normalization(std::initializer_list<unsigned char>{0});
        public:
            TransformerEncoderImpl(unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&) override;
        };
    }
}

