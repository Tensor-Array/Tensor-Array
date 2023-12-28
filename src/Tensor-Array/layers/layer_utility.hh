#include "layer_holder.hh"

#pragma once

namespace tensor_array
{
	namespace layers
	{
        typedef value::Tensor(*LayerFunction)(const value::Tensor&);

        class CUDA_ML_API ActivationImpl final :
            public TensorCalculateLayerImpl
        {
        public:
            ActivationImpl(LayerFunction);
            value::Tensor calculate(const value::Tensor&) override;
        private:
            const LayerFunction func;
        };

        class CUDA_ML_API ReShapeImpl final :
            public TensorCalculateLayerImpl
        {
        public:
            ReShapeImpl(const std::initializer_list<unsigned int>&);
            value::Tensor calculate(const value::Tensor&) override;
        private:
            const std::vector<unsigned int> shape;
        };

        value::Tensor CUDA_ML_API NoActivation(const value::Tensor&);
        value::Tensor CUDA_ML_API ReLU(const value::Tensor&);
        value::Tensor CUDA_ML_API tanh(const value::Tensor&);
        value::Tensor CUDA_ML_API Sigmoid(const value::Tensor&);
        value::Tensor CUDA_ML_API SoftMax(const value::Tensor&, unsigned char dim);

        using Activation = LayerHolder<ActivationImpl>;
        using ReShape = LayerHolder<ReShapeImpl>;
	}
}
