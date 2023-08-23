from transformers import PretrainedConfig
from typing import List


class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"
    def __init__(
        self,
        hidden_dim: int = 768, 
        n_layers: int = 12,
        n_heads: int = 12,
        seq_len: int = 2048,
        vocab_size: int = 50304,
        pre_ln: bool = False,
        pos_embed_type: str = "rope",
        weight_tying: bool = False,
        attn_type: str = 'xformers',
        apply_qk_norm: bool = False
        **kwargs
    ):
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pre_ln = pre_ln
        self.pos_embed_type = pos_embed_type
        self.weight_tying = weight_tying
        self.attn_type = attn_type
        self.apply_qk_norm = apply_qk_norm
        super().__init__(**kwargs)



if __name__ == '__main__':
    config = {
        "hidden_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "seq_len": 2048,
        "vocab_size": 50304,
        "pre_ln": False,
        "pos_embed_type": "rope",
        "weight_tying": False,
        "attn_type": "xformers"
    }
    openlm_config = OpenLMConfig(config)
    openlm_config.save_pretrained("open_lm_config")

