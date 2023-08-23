from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils.transformers.config import OpenLMConfig
from open_lm.model import ModelArgs, Transformer
import torch
from typing import Union, Tuple, Optional, List

def create_model(cfg):

    model_args = ModelArgs(
        dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        pre_ln=cfg.pre_ln,
        pos_embed_type=cfg.pos_embed_type,
        weight_tying=cfg.weight_tying,
        attn_type=cfg.attn_type,
        apply_qk_norm=cfg.apply_qk_norm
    )
    model = Transformer(model_args)

    return model


class OpenLMModel(PreTrainedModel):
    config_class = OpenLMConfig

    def __init__(self, config):
        super().__init__(config)
        print(config)
        self.model = create_model(config)

    def forward(self, tokens):
        return self.model(tokens)


class OpenLMforCausalLM(OpenLMModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = create_model(config)
        self.lm_head = None        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, OpenLlamaForCausalLM
        >>> model = OpenLlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        logits, hidden_states = self.model(input_ids, output_hidden_states=True)
        output = CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states,
        )
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


if __name__ == '__main__':
    openlm_config = OpenLMConfig.from_pretrained("utils/transformers/open_lm_config")
    print(OpenLMModel(openlm_config))
