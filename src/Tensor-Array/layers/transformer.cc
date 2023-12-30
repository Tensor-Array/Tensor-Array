#include "transformer.hh"
#include "layer_utility.hh"

namespace tensor_array
{
	namespace layers
	{
		TransformerEncoderImpl::TransformerEncoderImpl(unsigned int d_model, unsigned int n_head, unsigned int ff_size) :
			multihead_attn(d_model, n_head),
			feed_forward
			{
				Linear(ff_size),
				Activation(&ReLU),
				Linear(d_model)
			}
		{
			this->map_layer.insert(std::make_pair("MultiheadAttn", multihead_attn.get_shared()));
			this->map_layer.insert(std::make_pair("FeedForward", feed_forward.get_shared()));
			this->map_layer.insert(std::make_pair("LayerNorm1", layer_norm_1.get_shared()));
			this->map_layer.insert(std::make_pair("LayerNorm2", layer_norm_2.get_shared()));
		}
		value::Tensor TransformerEncoderImpl::calculate(const value::Tensor& input)
		{
			auto attn_output = this->multihead_attn(input, input, input);
			attn_output = this->layer_norm_1(input + attn_output);
			auto ff_output = this->feed_forward(attn_output);
			return this->layer_norm_2(attn_output + ff_output);
		}
		TransformerDecoderImpl::TransformerDecoderImpl(unsigned int d_model, unsigned int n_head, unsigned int ff_size) :
			masked_multihead_attn(d_model, n_head),
			multihead_attn(d_model, n_head),
			feed_forward
			{
				Linear(ff_size),
				Activation(&ReLU),
				Linear(d_model)
			}
		{
			this->map_layer.insert(std::make_pair("MaskedMultiheadAttn", masked_multihead_attn.get_shared()));
			this->map_layer.insert(std::make_pair("MultiheadAttn", multihead_attn.get_shared()));
			this->map_layer.insert(std::make_pair("FeedForward", feed_forward.get_shared()));
			this->map_layer.insert(std::make_pair("LayerNorm1", layer_norm_1.get_shared()));
			this->map_layer.insert(std::make_pair("LayerNorm2", layer_norm_2.get_shared()));
			this->map_layer.insert(std::make_pair("LayerNorm3", layer_norm_3.get_shared()));
		}
		value::Tensor TransformerDecoderImpl::calculate(const value::Tensor& tgt, const value::Tensor& memory_encode, const value::Tensor& tgt_mask)
		{
			auto attn_output = this->masked_multihead_attn(tgt, tgt, tgt, tgt_mask);
			attn_output = this->layer_norm_1(tgt + attn_output);

			auto attn_output_2 = this->multihead_attn(attn_output, memory_encode, memory_encode);
			attn_output = this->layer_norm_2(attn_output + attn_output_2);

			auto ff_output = this->feed_forward(attn_output_2);
			ff_output = this->layer_norm_3(attn_output_2 + ff_output);
			return ff_output;
		}
		TransformerImpl::TransformerImpl(unsigned int d_model, unsigned int n_head, unsigned int ff_size, unsigned int num_layers):
			fc(ff_size),
			encoder_blocks()
		{
			for (size_t i = 0; i < num_layers; i++)
			{
				encoder_blocks->insert(TransformerEncoder(d_model, n_head, ff_size));
				TransformerDecoder temp_decoder(d_model, n_head, ff_size);
				decoder_blocks.push_back(std::move(temp_decoder));
			}
		}
		value::Tensor TransformerImpl::calculate(const value::Tensor& src, const value::Tensor& tgt, const value::Tensor& tgt_mask)
		{
			value::Tensor src_e = src;
			value::Tensor tgt_e = tgt;
			src_e = this->encoder_blocks(src_e);
			for (auto& it : this->decoder_blocks)
				tgt_e = it(tgt_e, src_e, tgt_mask);
			return SoftMax(this->fc(tgt_e), 0);
		}
	}
}
