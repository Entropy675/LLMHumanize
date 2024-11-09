import torch.nn as nn
from transformers import BertModel  # Import a pre-trained BERT model for external knowledge

class CuriousAttention(nn.Module):
    def __init__(self):
        super(CuriousAttention, self).__init__()
        if bert_model is None:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert = bert_model

    def forward(self, query, key):
        # Compute the attention weights using softmax
        attention_weights = torch.softmax(torch.matmul(query, key.T) / math.sqrt(query.shape[-1]), dim=-1)

        # Using external knowledge to estimate novelty scores...
        external_knowledge = self.bert(key)
        novelty_scores = (1 - attention_weights.mean(dim=-1)) * (-external_knowledge.mean(dim=-1))

        return novelty_scores
