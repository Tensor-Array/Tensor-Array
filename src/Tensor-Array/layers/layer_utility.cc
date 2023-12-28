#include "layer_utility.hh"

namespace tensor_array
{
    namespace layers
    {
        ActivationImpl::ActivationImpl(LayerFunction func) :
            func(func)
        {
        }

        value::Tensor ActivationImpl::calculate(const value::Tensor& input)
        {
            return this->func(input);
        }

        ReShapeImpl::ReShapeImpl(const std::initializer_list<unsigned int>& shape) :
            shape(shape)
        {
        }

        value::Tensor ReShapeImpl::calculate(const value::Tensor& input)
        {
            std::vector<unsigned int> shape = this->shape;
            shape.insert(shape.begin(), input.get_buffer().shape().begin()[0]);
            return input.reshape(shape);
        }

        value::Tensor NoActivation(const value::Tensor& input)
        {
            return input;
        }

        value::Tensor Sigmoid(const value::Tensor& input)
        {
            return input.sigmoid();
        }
        value::Tensor ReLU(const value::Tensor& input)
        {
            value::Tensor temp_zeros = value::values(input.get_buffer().shape(), 0.f);
            return condition(input > temp_zeros, input, temp_zeros);
        }

        value::Tensor tanh(const value::Tensor& input)
        {
            return input.tanh();
        }

        value::Tensor SoftMax(const value::Tensor& input, unsigned char dim)
        {
            if (dim)
            {
                std::vector<value::Tensor> temp;
                for (value::Tensor ten : input)
                    temp.push_back(SoftMax(ten, dim - 1));
                return add_dim(temp);
            }
            value::Tensor a1 = input.exp();
            value::Tensor a2 = a1.value_scalar();
            return a1 / a2;
        }
    }
}