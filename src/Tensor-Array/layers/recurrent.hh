#include "linear.hh"
#include "layer_utility.hh"
#pragma once

namespace tensor_array
{
	namespace layers
	{
        class CUDA_ML_API RecurrentImpl :
            public TensorCalculateLayerImpl
        {
        private:
            value::Tensor hidden;
            Linear layer_input, layer_hidden;
            LayerFunction act;
            bool copy_after_calculate = false;
        public:
            RecurrentImpl(const value::Tensor&, LayerFunction);
            RecurrentImpl(const value::Tensor&);
            RecurrentImpl(unsigned int, LayerFunction);
            value::Tensor calculate(const value::Tensor&) override;
            void change_hidden_value(const value::Tensor&);
            void set_copy_after_calculate(bool);
            friend class LSTM_Impl;
        };
        using Recurrent = LayerHolder<RecurrentImpl>;

        class CUDA_ML_API LSTM_Impl :
            public TensorCalculateLayerImpl
        {
        private:
            value::Tensor lstm_cell;
            Recurrent gate_in, gate_forget, gate_cell, gate_out;
            bool copy_after_calculate = false;
        public:
            LSTM_Impl(const value::Tensor&);
            LSTM_Impl(unsigned int);
            value::Tensor calculate(const value::Tensor&) override;
            void change_hidden_value(const value::Tensor&, const value::Tensor&);
            void set_copy_after_calculate(bool);
            const value::Tensor& get_cell() const;
        };
		using LSTM = LayerHolder<LSTM_Impl>;
	}
}

