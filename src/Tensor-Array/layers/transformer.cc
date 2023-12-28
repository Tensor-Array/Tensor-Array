#include "transformer.hh"
#include "layer_utility.hh"

namespace tensor_array
{
	namespace layers
	{
		MultiHeadAttentionImpl::MultiHeadAttentionImpl(unsigned int d_model, unsigned int n_head):
			w_q(d_model),
			w_k(d_model),
			w_v(d_model),
			w_o(d_model),
			d_model(d_model),
			n_head(n_head)
		{
		}

		value::Tensor scaled_dot_product_attention(const value::Tensor& q, const value::Tensor& k, const value::Tensor& v)
		{
			auto attn_scores = value::matmul(q, k.transpose(k.get_buffer().shape().size() - 2, k.get_buffer().shape().size() - 1));
			auto attn_probs = SoftMax(attn_scores, attn_scores.get_buffer().shape().size() - 1);
			return matmul(attn_probs, v);
		}

		value::Tensor MultiHeadAttentionImpl::calculate(const value::Tensor& input_q, const value::Tensor& input_k, const value::Tensor& input_v)
		{
			value::Tensor temp_q = this->w_q(input_q);
			value::Tensor temp_k = this->w_q(input_k);
			value::Tensor temp_v = this->w_q(input_v);
			temp_q = temp_q.reshape({ temp_q.get_buffer().shape().begin()[0], temp_q.get_buffer().shape().end()[-1], this->n_head, this->d_model / this->n_head }).transpose(1, 2);
			temp_k = temp_k.reshape({ temp_k.get_buffer().shape().begin()[0], temp_k.get_buffer().shape().end()[-1], this->n_head, this->d_model / this->n_head }).transpose(1, 2);
			temp_v = temp_v.reshape({ temp_v.get_buffer().shape().begin()[0], temp_v.get_buffer().shape().end()[-1], this->n_head, this->d_model / this->n_head }).transpose(1, 2);

			auto attention_output = scaled_dot_product_attention(temp_q, temp_k, temp_v);

			attention_output = attention_output.transpose(1, 2).reshape({ attention_output.get_buffer().shape().begin()[0], this->d_model, this->d_model });

			return this->w_o(attention_output);
		}
		TransformerEncoderImpl::TransformerEncoderImpl(unsigned int d_model, unsigned int n_head):
			multihead_attn(d_model, n_head),
			feed_forward
			{
				Linear(4 * d_model),
				Activation(&ReLU),
				Linear(d_model)
			}
		{
		}
		value::Tensor TransformerEncoderImpl::calculate(const value::Tensor& input)
		{
			auto attn_output = this->multihead_attn(input, input, input);
			attn_output = this->layer_norm_1(input + attn_output);
			auto ff_output = this->feed_forward(attn_output);
			return this->layer_norm_2(attn_output + ff_output);
		}
	}
}
