from transformers import OPTForCausalLM
from torch import nn

class OPTForCausalLM_ours(OPTForCausalLM):
    def __init__(self, config, d_latent):
        super().__init__(config)

        # z_dim to hidden
        self.z_to_hidden = nn.Linear(d_latent, config.hidden_size)
        self.hidden_to_z = nn.Linear(config.hidden_size, 2*d_latent)

        # Initialize weights and apply final processing
        self.post_init()
