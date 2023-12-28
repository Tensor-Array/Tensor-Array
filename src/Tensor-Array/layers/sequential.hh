#include "layer_holder.hh"
#pragma once

namespace tensor_array
{
	namespace layers
	{
        using LayerInSequential = LayerHolder<TensorCalculateLayerImpl>;

        class CUDA_ML_API SequentialImpl final :
            public TensorCalculateLayerImpl
        {
        private:
            std::vector<LayerInSequential> function_list;
        public:
            SequentialImpl() = default;
            SequentialImpl(const std::initializer_list<LayerInSequential>&);
            void insert(const LayerInSequential&);
            void insert(LayerInSequential&&);
            LayerInSequential& get(std::size_t);
            value::Tensor calculate(const value::Tensor&) override;
        };

        class CUDA_ML_API Sequential : public LayerHolder<SequentialImpl>
        {
        public:
            Sequential() = default;
            Sequential(std::initializer_list<LayerInSequential>&& list_holder);
        };
	}
}

