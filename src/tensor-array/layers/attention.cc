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

#include "attention.hh"
#include "layer_utility.hh"
#include <cmath>

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
			this->map_layer.insert(std::make_pair("linear_q", w_q.get_shared()));
			this->map_layer.insert(std::make_pair("linear_k", w_k.get_shared()));
			this->map_layer.insert(std::make_pair("linear_v", w_v.get_shared()));
			this->map_layer.insert(std::make_pair("linear_0", w_o.get_shared()));
		}

		void MultiHeadAttentionImpl::layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&& vector_shape)
		{
		}

		value::Tensor scaled_dot_product_attention(const value::Tensor& q, const value::Tensor& k, const value::Tensor& v, const value::Tensor& mask)
		{
			auto attn_scores = value::matmul(q, k.transpose(k.get_buffer().shape().size() - 2, k.get_buffer().shape().size() - 1));
			if (mask.has_tensor())
				attn_scores = value::condition(mask, attn_scores, value::values(attn_scores.get_buffer().shape(), -INFINITY));
			auto attn_probs = SoftMax(attn_scores, attn_scores.get_buffer().shape().size() - 1);
			return matmul(attn_probs, v);
		}

		value::Tensor MultiHeadAttentionImpl::calculate(const value::Tensor& input_q, const value::Tensor& input_k, const value::Tensor& input_v, const value::Tensor &mask)
		{
			value::Tensor temp_q = this->w_q(input_q);
			value::Tensor temp_k = this->w_k(input_k);
			value::Tensor temp_v = this->w_v(input_v);
			temp_q = temp_q.reshape({ temp_q.get_buffer().shape().begin()[0], temp_q.get_buffer().shape().begin()[1], this->n_head, this->d_model / this->n_head }).transpose(1, 2);
			temp_k = temp_k.reshape({ temp_k.get_buffer().shape().begin()[0], temp_k.get_buffer().shape().begin()[1], this->n_head, this->d_model / this->n_head }).transpose(1, 2);
			temp_v = temp_v.reshape({ temp_v.get_buffer().shape().begin()[0], temp_v.get_buffer().shape().begin()[1], this->n_head, this->d_model / this->n_head }).transpose(1, 2);

			value::Tensor attention_output = scaled_dot_product_attention(temp_q, temp_k, temp_v, mask);

			attention_output = attention_output.transpose(1, 2);
			attention_output = attention_output.reshape({ attention_output.get_buffer().shape().begin()[0], attention_output.get_buffer().shape().begin()[1], this->d_model });

			return this->w_o(attention_output);
		}
	}
}
