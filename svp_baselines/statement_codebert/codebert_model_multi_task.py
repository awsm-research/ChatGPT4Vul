import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.Dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, 1)
        
        self.rnn_pool = nn.GRU(input_size=768,
                                hidden_size=768,
                                num_layers=1,
                                batch_first=True)
        self.func_dense = nn.Linear(hidden_dim, hidden_dim)
        self.func_out_proj = nn.Linear(hidden_dim, 2)
        
    def forward(self, hidden):
        x = self.Dropout(hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)
        
        out, func_x = self.rnn_pool(hidden)
        func_x = func_x.squeeze(0)
        func_x = self.Dropout(func_x)
        func_x = self.func_dense(func_x)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x

class Model(nn.Module):   
    def __init__(self, roberta, tokenizer, args, hidden_dim=768, num_labels=155):
        super(Model, self).__init__()
        self.word_embedding = roberta.embeddings.word_embeddings
        self.rnn_statement_embedding = nn.GRU(input_size=768,
                                              hidden_size=768,
                                              num_layers=1,
                                              batch_first=True)
        self.roberta = roberta
        self.tokenizer = tokenizer
        self.args = args
        # CLS head
        self.classifier = ClassificationHead(hidden_dim=hidden_dim)

    def forward(self, input_ids_with_pattern, statement_mask, total_training_samples=None, labels=None, func_labels=None, step=None):
        statement_mask = statement_mask[:, :self.args.num_labels]
        if self.training:
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.randn(embed.shape[0], embed.shape[1], embed.shape[3]).to(self.args.device)
            for i in range(len(embed)):
                statement_of_tokens = embed[i]
                out, statement_embed = self.rnn_statement_embedding(statement_of_tokens)
                inputs_embeds[i, :, :] = statement_embed
            inputs_embeds = inputs_embeds[:, :self.args.num_labels, :]
            rep = self.roberta(inputs_embeds=inputs_embeds, attention_mask=statement_mask).last_hidden_state
            
            logits, func_logits = self.classifier(rep)
            
            loss_fct = nn.CrossEntropyLoss()
            statement_loss = loss_fct(logits, labels)
            loss_fct_2 = nn.CrossEntropyLoss()
            func_loss = loss_fct_2(func_logits, func_labels)
            return statement_loss, func_loss
        else:
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.randn(embed.shape[0], embed.shape[1], embed.shape[3]).to(self.args.device)
            for i in range(len(embed)):
                statement_of_tokens = embed[i]
                out, statement_embed = self.rnn_statement_embedding(statement_of_tokens)
                inputs_embeds[i, :, :] = statement_embed
            inputs_embeds = inputs_embeds[:, :self.args.num_labels, :]
            rep = self.roberta(inputs_embeds=inputs_embeds, attention_mask=statement_mask).last_hidden_state
            logits, func_logits = self.classifier(rep)
            
            probs = torch.sigmoid(logits)
            func_probs = torch.softmax(func_logits, dim=-1)
            return probs, func_probs