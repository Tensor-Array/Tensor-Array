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
