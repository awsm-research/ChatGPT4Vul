import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.Dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, 155)
        self.func_dense = nn.Linear(hidden_dim, hidden_dim)
        self.func_out_proj = nn.Linear(hidden_dim, 2)
        
    def forward(self, hidden):
        hidden = hidden[:, 0, :]
        hidden = self.Dropout(hidden)
        x = self.dense(hidden)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)
        
        func_x = self.func_dense(hidden)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x

class Model(nn.Module):   
    def __init__(self, roberta, tokenizer, args, hidden_dim=768):
        super(Model, self).__init__()
        self.roberta = roberta
        self.tokenizer = tokenizer
        self.args = args
        # CLS head
        self.classifier = ClassificationHead(hidden_dim=hidden_dim)

    def forward(self, source_ids, position_idx, attn_mask, labels=None, func_labels=None):
        #embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.roberta.embeddings.word_embeddings(source_ids)
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:,:,None] + avg_embeddings*nodes_mask[:,:,None]
        rep = self.roberta(inputs_embeds=inputs_embeddings, attention_mask=source_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        if self.training:
            logits, func_logits = self.classifier(rep)
            loss_fct = nn.CrossEntropyLoss()
            statement_loss = loss_fct(logits, labels)
            loss_fct_2 = nn.CrossEntropyLoss()
            func_loss = loss_fct_2(func_logits, func_labels)
            return statement_loss, func_loss
        else:
            logits, func_logits = self.classifier(rep)
            probs = torch.sigmoid(logits)
            func_probs = torch.softmax(func_logits, dim=-1)
            return probs, func_probs