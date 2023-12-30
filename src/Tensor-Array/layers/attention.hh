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
            public LayerImpl,
            public CalculateStruct<value::Tensor, const value::Tensor&, const value::Tensor&, const value::Tensor&, const value::Tensor&>
        {
        private:
            const unsigned int d_model, n_head;
            Linear w_q, w_k, w_v, w_o;
        public:
            MultiHeadAttentionImpl(unsigned int, unsigned int);
            void init_value(const value::Tensor&, const value::Tensor&, const value::Tensor&, const value::Tensor& = value::Tensor()) override;
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&, const value::Tensor& = value::Tensor()) override;
        };

        using MultiHeadAttention = LayerHolder<MultiHeadAttentionImpl>;
    }
}