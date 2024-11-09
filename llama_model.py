import torch
from transformers import LLaMModel, LLaMTokenizer
from curious_attention import CuriousAttention

class CuriousLLaMModel(LLaMModel):
    def __init__(self, *args, **kwargs):
        super(CuriousLLaMModel, self).__init__(*args, **kwargs)
        self.novelty_attention = CuriousAttention()

    def compute_attention(self, query, key, value):
        # Compute the original attention weights using softmax
        attention_weights = torch.softmax(torch.matmul(query, key.T) / math.sqrt(query.shape[-1]), dim=-1)

        # Compute the novelty scores using the custom layer
        novelty_scores = self.novelty_attention(novelty_attention.query, attention_weights)

        # Compute the final attention weights by adding the novelty scores
        final_attention_weights = (1 - novelty_scores.mean(dim=-1)) * attention_weights

        return final_attention_weights

