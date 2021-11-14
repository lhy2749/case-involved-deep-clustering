import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
import math
import gcn


class SeqEncoder(nn.Module):
    """
    context information extractor
    """

    def __init__(self, vocab_size, seq_mid_hid, seq_len, num_heads, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, seq_mid_hid, padding_idx=0)
        self.position_embeddings = nn.Embedding(seq_len, seq_mid_hid)
        self.self_attention = SelfAttention(seq_mid_hid, num_heads, dropout)
        self.LayerNorm = nn.LayerNorm(seq_mid_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, mask, device):
        seq_len = x_seq.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).to(device)
        position_ids = position_ids.unsqueeze(0).expand_as(x_seq)
        words_embeddings = self.word_embeddings(x_seq)
        # words_embeddings = self.dropout(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        # words_embeddings = self.dropout(words_embeddings)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        context_layer, attention_probs = self.self_attention(embeddings, mask)
        return context_layer[:, 0, :], context_layer, attention_probs


class SelfAttention(nn.Module):
    """
    context sequence attention
    """

    def __init__(self, seq_mid_hid, num_heads, dropout):
        super().__init__()
        if seq_mid_hid % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (seq_mid_hid, num_heads))
        self.output_attentions = True
        self.num_attention_heads = num_heads
        self.attention_head_size = int(seq_mid_hid / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(seq_mid_hid, self.all_head_size)
        self.key = nn.Linear(seq_mid_hid, self.all_head_size)
        self.value = nn.Linear(seq_mid_hid, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0
        attention_scores = attention_scores * extended_attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class VaeModel(nn.Module):
    def __init__(self, vae_mid,
                 num_words, vocab_size, bow_mid_hid,
                 seq_mid_hid, seq_len, num_heads,
                 dropout, is_traing, support):
        super().__init__()
        self.training = is_traing
        assert (bow_mid_hid == seq_mid_hid)
        self.seq_en = SeqEncoder(vocab_size, seq_mid_hid, seq_len, num_heads, dropout)
        self.mu_line = nn.Linear(bow_mid_hid * 2, vae_mid)
        self.var_line = nn.Linear(bow_mid_hid * 2, vae_mid)
        self.gcn_layer = gcn.GCN(input_dim=12652, support=support, num_classes=512)

    def reparameterize(self, mu, var):
        if self.training:
            std = torch.exp(var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x_bow, x_seq, mask, device, xh, t_features):
        gcn_out = self.gcn_layer(t_features)
        gcn_out = gcn_exc(gcn_out, xh)
        gcn_out = gcn_out.to(device)
        x_seq, _, _ = self.seq_en(x_seq, mask, device)
        x = torch.cat([x_seq, gcn_out], dim=-1)
        mu = self.mu_line(x)
        var = self.var_line(x)
        z = self.reparameterize(mu, var)
        return x, mu, var, z


class DCTC(nn.Module):
    def __init__(self, vae_mid,
                 num_words, vocab_size, bow_mid_hid,
                 seq_mid_hid, seq_len, num_heads,
                 dropout, is_traing, support):
        super().__init__()
        self.vae = VaeModel(vae_mid, num_words, vocab_size, bow_mid_hid, seq_mid_hid, seq_len, num_heads, dropout,
                            is_traing, support)
        self.fc_bow_de = nn.Linear(vae_mid, num_words)
        torch.nn.init.kaiming_normal_(self.fc_bow_de.weight)
        self.cluster_layer = Parameter(torch.Tensor(15, vae_mid))

    def forward(self, x_bow, x_seq, mask, device, xh, t_features):
        x, mu, var, z = self.vae(x_bow, x_seq, mask, device, xh, t_features)
        thate = F.softmax(z, dim=-1)
        out = self.fc_bow_de(thate)
        q = 1.0 / (1.0 + torch.sum(torch.pow(mu.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return out, mu, var, z, x, q


def gcn_exc(gcn_out, xh):
    a = xh.shape[0]
    b = gcn_out.shape[1]
    result = torch.zeros((a, b))
    for i in range(a):
        result[i] = gcn_out[xh[i]]
    return result
