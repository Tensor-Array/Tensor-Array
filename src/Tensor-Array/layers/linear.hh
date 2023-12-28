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
            void init_value(const value::Tensor& input) override;
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


