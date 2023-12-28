#include "layer_holder.hh"
#pragma once

namespace tensor_array
{
	namespace layers
	{
        class CUDA_ML_API NormalizationImpl final :
            public TensorCalculateLayerImpl
        {
        private:
            const float eps, momentum;
            const std::vector<unsigned char> dims_mean;
            value::Tensor moving_mean, moving_variance, gamma, beta;
        public:
            NormalizationImpl(const std::initializer_list<unsigned char>&, float = 1e-5f, float = .1f);
            ~NormalizationImpl();
            void init_value(const value::Tensor&) override;
            value::Tensor calculate(const value::Tensor&) override;
        };

        using Normalization = LayerHolder<NormalizationImpl>;
	}
}

