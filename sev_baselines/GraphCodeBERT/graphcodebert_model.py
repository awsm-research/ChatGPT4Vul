# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from focal_loss import FocalLoss 


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    def __init__(self, encoder, tokenizer, config, args):
        super(Model, self).__init__()
        self.classifier = RobertaClassificationHead(config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config=config
        self.args = args
        
    def forward(self, source_ids, position_idx, attn_mask, labels=None):   
        #embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.encoder.embeddings.word_embeddings(source_ids)
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:,:,None] + avg_embeddings*nodes_mask[:,:,None]  
        
        last_hidden_state = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx).last_hidden_state
        logits = self.classifier(last_hidden_state)
        if labels is not None:
            reg_loss_fct = nn.MSELoss()
            labels = labels.unsqueeze(-1)
            cvss_loss = reg_loss_fct(logits, labels)
            return cvss_loss
        else:
            return logits
