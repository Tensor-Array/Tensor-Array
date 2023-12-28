#include "recurrent.hh"

namespace tensor_array
{
	namespace layers
	{
        value::Tensor RecurrentImpl::calculate(const value::Tensor& input)
        {
            value::Tensor temp = this->act(add(this->layer_input(input), this->layer_hidden(this->hidden)));
            this->change_hidden_value(temp);
            return temp;
        }

        RecurrentImpl::RecurrentImpl(const value::Tensor& start, LayerFunction act) :
            hidden(start),
            layer_input(start.get_buffer().shape().end()[-1]),
            layer_hidden(start.get_buffer().shape().end()[-1]),
            act(act)
        {
            this->map_layer.insert(std::make_pair("input", layer_input.get_shared()));
            this->map_layer.insert(std::make_pair("hidden", layer_hidden.get_shared()));
        }

        RecurrentImpl::RecurrentImpl(const value::Tensor& start) :
            RecurrentImpl(start, NoActivation)
        {
        }

        RecurrentImpl::RecurrentImpl(unsigned int list_dim_size, LayerFunction act) :
            RecurrentImpl(value::values({ 1, list_dim_size }, 0), act)
        {
        }

        void RecurrentImpl::change_hidden_value(const value::Tensor& hidden_value)
        {
            this->hidden = hidden_value;
            if (this->copy_after_calculate)
                this->hidden = this->hidden.new_grad_copy();
        }

        void RecurrentImpl::set_copy_after_calculate(bool copy_after_calculate)
        {
            this->copy_after_calculate = copy_after_calculate;
        }

        value::Tensor LSTM_Impl::calculate(const value::Tensor& input)
        {
            value::Tensor temp_cell = add(multiply(this->gate_forget(input), this->lstm_cell), multiply(this->gate_in(input), this->gate_cell(input)));
            value::Tensor temp_hidden = multiply(this->gate_out(input), tanh(temp_cell));
            this->change_hidden_value(temp_hidden, temp_cell);
            return temp_hidden;
        }
        LSTM_Impl::LSTM_Impl(const value::Tensor& start_hidden) :
            lstm_cell(value::values({}, 0.f)),
            gate_in(start_hidden, Sigmoid),
            gate_forget(start_hidden, Sigmoid),
            gate_cell(start_hidden, tanh),
            gate_out(start_hidden, Sigmoid)
        {
            this->map_layer.insert(std::make_pair("gate_in", gate_in.get_shared()));
            this->map_layer.insert(std::make_pair("gate_forget", gate_forget.get_shared()));
            this->map_layer.insert(std::make_pair("gate_cell", gate_cell.get_shared()));
            this->map_layer.insert(std::make_pair("gate_out", gate_out.get_shared()));
        }

        LSTM_Impl::LSTM_Impl(unsigned int list_dim_size) :
            LSTM_Impl(value::values({ 1, list_dim_size }, 0.f))
        {}

        void LSTM_Impl::change_hidden_value(const value::Tensor& hidden_value, const value::Tensor& cell_value)
        {
            value::Tensor temp_copy = hidden_value;
            if (hidden_value.has_tensor() && this->copy_after_calculate)
                temp_copy = temp_copy.new_grad_copy();
            this->gate_forget.get()->hidden = temp_copy;
            this->gate_in.get()->hidden = temp_copy;
            this->gate_cell.get()->hidden = temp_copy;
            this->gate_out.get()->hidden = temp_copy;
            this->lstm_cell = cell_value;
            if (cell_value.has_tensor() && this->copy_after_calculate)
                this->lstm_cell = this->lstm_cell.new_grad_copy();
        }

        void LSTM_Impl::set_copy_after_calculate(bool copy_after_calculate)
        {
            this->copy_after_calculate = copy_after_calculate;
            this->gate_forget->copy_after_calculate = copy_after_calculate;
            this->gate_in->copy_after_calculate = copy_after_calculate;
            this->gate_cell->copy_after_calculate = copy_after_calculate;
            this->gate_out->copy_after_calculate = copy_after_calculate;
        }

        const value::Tensor& LSTM_Impl::get_cell() const
        {
            return this->lstm_cell;
        }
	}
}