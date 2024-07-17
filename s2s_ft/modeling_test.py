# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss

class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None,
                 reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)

logger = logging.getLogger(__name__)
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 new_pos_ids=False,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 ffn_type=0,
                 label_smoothing=None,
                 num_qkv=0,
                 seg_emb=False,
                 source_type_id=0,
                 target_type_id=1,
                 no_segment_embedding=False, **kwargs):
        """Constructs BertConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.new_pos_ids = new_pos_ids
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.ffn_type = ffn_type
            self.label_smoothing = label_smoothing
            self.num_qkv = num_qkv
            self.seg_emb = seg_emb
            self.no_segment_embedding = no_segment_embedding
            self.source_type_id = source_type_id
            self.target_type_id = target_type_id
            if type_vocab_size == 0:
                self.no_segment_embedding = True
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file, **kwargs):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        json_info = json.loads(text)
        for k, v in kwargs.items():
            json_info[k] = v
        return cls.from_dict(json_info)

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class LayoutlmEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(LayoutlmEmbeddings, self).__init__()
       
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size )
        """Pad 0 Text 1 PIC 2 Title 3 sep 6 """
        self.label_type_embeddings = nn.Embedding(10, config.hidden_size)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,bbox, token_type_ids=None, position_ids=None,input_label_data=None, task_idx=None):
        position_embeddings = self.position_embeddings(position_ids) #1 n
        if bbox.size(2)<8:
            """layout"""
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
            h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
            w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
            embeddings = (
                    position_embeddings             
                    + left_position_embeddings
                    + upper_position_embeddings
                    + right_position_embeddings
                    + lower_position_embeddings
                    + h_position_embeddings
                    + w_position_embeddings
            )
        else:
            """box"""
            left_up_x_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            left_up_y_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_up_x_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            right_up_y_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
            right_down_x_position_embeddings = self.x_position_embeddings(bbox[:, :, 4])
            right_down_y_position_embeddings = self.y_position_embeddings(bbox[:, :, 5])
            left_down_x_position_embeddings = self.x_position_embeddings(bbox[:, :, 6])
            left_down_y_position_embeddings = self.y_position_embeddings(bbox[:, :, 7])
            
            embeddings = (
                    left_up_x_position_embeddings
                    + left_up_y_position_embeddings
                    + right_up_x_position_embeddings
                    + right_up_y_position_embeddings
                    + right_down_x_position_embeddings
                    + right_down_y_position_embeddings
                    + left_down_x_position_embeddings
                    + left_down_y_position_embeddings
                    + position_embeddings) 
        
        if input_label_data is not None:
            label_embeddings=self.label_type_embeddings(input_label_data)
            embeddings = embeddings + label_embeddings 
        embeddings = embeddings + self.token_type_embeddings(token_type_ids) 
           
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if hasattr(config, 'num_qkv') and (config.num_qkv > 1):
            self.num_qkv = config.num_qkv
        else:
            self.num_qkv = 1

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size * self.num_qkv)
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size * self.num_qkv)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.uni_debug_flag = True if os.getenv(
            'UNI_DEBUG_FLAG', '') else False
        if self.uni_debug_flag:
            self.register_buffer('debug_attention_probs',
                                 torch.zeros((512, 512)))
        if hasattr(config, 'seg_emb') and config.seg_emb:
            self.b_q_s = nn.Parameter(torch.zeros(
                1, self.num_attention_heads, 1, self.attention_head_size))
            self.seg_emb = nn.Embedding(
                config.type_vocab_size, self.all_head_size)
        else:
            self.b_q_s = None
            self.seg_emb = None

    def transpose_for_scores(self, x, mask_qkv=None):
        if self.num_qkv > 1:
            sz = x.size()[:-1] + (self.num_qkv,
                                  self.num_attention_heads, self.all_head_size)
            # (batch, pos, num_qkv, head, head_hid)
            x = x.view(*sz)
            if mask_qkv is None:
                x = x[:, :, 0, :, :]
            elif isinstance(mask_qkv, int):
                x = x[:, :, mask_qkv, :, :]
            else:
                # mask_qkv: (batch, pos)
                if mask_qkv.size(1) > sz[1]:
                    mask_qkv = mask_qkv[:, :sz[1]]
                # -> x: (batch, pos, head, head_hid)
                x = x.gather(2, mask_qkv.view(sz[0], sz[1], 1, 1, 1).expand(
                    sz[0], sz[1], 1, sz[3], sz[4])).squeeze(2)
        else:
            sz = x.size()[:-1] + (self.num_attention_heads,
                                  self.attention_head_size)
            # (batch, pos, head, head_hid)
            x = x.view(*sz)
        # (batch, head, pos, head_hid)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None,
                mask_qkv=None, seg_ids=None, key_history=None, value_history=None,
                key_cache=None, value_cache=None,
                ):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(hidden_states, self.key.weight)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(x_states, self.key.weight)
            mixed_value_layer = self.value(x_states)

        if key_cache is not None and isinstance(key_cache, list):
            key_cache.append(mixed_key_layer  )
            mixed_key_layer = torch.cat(key_cache, dim=1)

        if value_cache is not None and isinstance(value_cache, list):
            value_cache.append(mixed_value_layer)
            mixed_value_layer = torch.cat(value_cache, dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer, mask_qkv)
        key_layer = self.transpose_for_scores(mixed_key_layer, mask_qkv)
        value_layer = self.transpose_for_scores(mixed_value_layer, mask_qkv)

        if key_history is not None and not isinstance(key_history, list):
            key_layer = torch.cat((key_history, key_layer), dim=-2)
            value_layer = torch.cat((value_history, value_layer), dim=-2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if self.seg_emb is not None:
            seg_rep = self.seg_emb(seg_ids)
            # (batch, pos, head, head_hid)
            seg_rep = seg_rep.view(seg_rep.size(0), seg_rep.size(
                1), self.num_attention_heads, self.attention_head_size)
            qs = torch.einsum('bnih,bjnh->bnij',
                              query_layer + self.b_q_s, seg_rep)
            attention_scores = attention_scores + qs

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.uni_debug_flag:
            _pos = attention_probs.size(-1)
            self.debug_attention_probs[:_pos, :_pos].copy_(
                attention_probs[0].mean(0).view(_pos, _pos))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if isinstance(key_history, list):
            key_history.append(key_layer)
        if isinstance(value_history, list):
            value_history.append(value_layer)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None,
                mask_qkv=None, seg_ids=None, key_history=None, value_history=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states,
            mask_qkv=mask_qkv, seg_ids=seg_ids, key_history=key_history, value_history=value_history)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TransformerFFN(nn.Module):
    def __init__(self, config):
        super(TransformerFFN, self).__init__()
        self.ffn_type = config.ffn_type
        assert self.ffn_type in (1, 2)
        if self.ffn_type in (1, 2):
            self.wx0 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (2,):
            self.wx1 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (1, 2):
            self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        if self.ffn_type in (1, 2):
            x0 = self.wx0(x)
            if self.ffn_type == 1:
                x1 = x
            elif self.ffn_type == 2:
                x1 = self.wx1(x)
            out = self.output(x0 * x1)
        out = self.dropout(out)
        out = self.LayerNorm(out + x)
        return out

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.ffn_type = config.ffn_type
        if self.ffn_type:
            self.ffn = TransformerFFN(config)
        else:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None,
                mask_qkv=None, seg_ids=None, key_history=None, value_history=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states,
            mask_qkv=mask_qkv, seg_ids=seg_ids, key_history=key_history, value_history=value_history)
        if self.ffn_type:
            layer_output = self.ffn(attention_output)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])   #12

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None, mask_qkv=None, seg_ids=None, key_history=None, value_history=None):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)
        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for i, layer_module in enumerate(self.layer):
                set_key = None
                if isinstance(key_history, list):
                    set_key = key_history if len(key_history) < len(self.layer) else key_history[i]
                set_value = None
                if isinstance(value_history, list):
                    set_value = value_history if len(key_history) < len(self.layer) else value_history[i]
                hidden_states = layer_module(
                    hidden_states, attention_mask, mask_qkv=mask_qkv, seg_ids=seg_ids,
                    key_history=set_key, value_history=set_value)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TextLineSPLMPredictionHead(nn.Module):
    def __init__(self, config, src_len):
        super(TextLineSPLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.bias = nn.Parameter(torch.zeros(src_len))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, src_emb):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        hidden_states = torch.einsum('btf,bsf->bts', hidden_states, src_emb) + self.bias
        return hidden_states
    
class TextLineSPLMPredictionHeadWithSelfAttention(nn.Module):
    def __init__(self, config, src_len):
        super(TextLineSPLMPredictionHeadWithSelfAttention, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.bias = nn.Parameter(torch.zeros(src_len))
        self.self_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)  # 自注意力机制
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, src_emb):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        src_emb = src_emb.transpose(0, 1).contiguous()
        src_emb, _ = self.self_attention(src_emb, src_emb, src_emb) 
        src_emb = src_emb.transpose(0, 1).contiguous() 
        hidden_states = torch.einsum('btf,bsf->bts', hidden_states, src_emb) + self.bias
        return hidden_states

class TextLineSPPreTrainingHeads(nn.Module):
    def __init__(self, config, src_len, num_labels=2):
        super(TextLineSPPreTrainingHeads, self).__init__()
        self.predictions = TextLineSPLMPredictionHeadWithSelfAttention(config, src_len)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, src_emb):
        prediction_scores = self.predictions(sequence_output, src_emb)
        if pooled_output is None:
            seq_relationship_score = None   #this
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # module.weight.data.copy_(torch.Tensor(
            #     truncnorm.rvs(-1, 1, size=list(module.weight.data.shape)) * self.config.initializer_range))
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, config, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in ('config_path', 'type_vocab_size', 'relax_projection', 'new_pos_ids', 'task_idx',
                          'max_position_embeddings', 'fp32_embedding', 'ffn_type', 'label_smoothing',
                          'hidden_dropout_prob', 'attention_probs_dropout_prob', 'num_qkv', 'seg_emb',
                          'word_emb_map', 'num_labels', 'num_rel', 'num_sentlvl_labels'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(pretrained_model_name, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        return model

class LayoutlmModel(PreTrainedBertModel):
    def __init__(self, config):
        super(LayoutlmModel, self).__init__(config)
        self.embeddings = LayoutlmEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def rescale_some_parameters(self):
        for layer_id, layer in enumerate(self.encoder.layer):
            layer.attention.output.dense.weight.data.div_(
                math.sqrt(2.0 * (layer_id + 1)))
            layer.output.dense.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def get_extended_attention_mask(self,attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                mask_qkv=None, task_idx=None, key_history=None, value_history=None, position_ids=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
                        input_ids[:, :, 1:], token_type_ids, task_idx=task_idx, position_ids=position_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      mask_qkv=mask_qkv, seg_ids=token_type_ids,
                                      key_history=key_history, value_history=value_history)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class TextLineModelIncr(LayoutlmModel):
    def __init__(self, config):
        super(TextLineModelIncr, self).__init__(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask,input_label_data=None, output_all_encoded_layers=True,
                prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, task_idx=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids[:, :, :], token_type_ids, position_ids,input_label_data,task_idx=task_idx)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv,
                                      seg_ids=token_type_ids)  #encode_layer include 12 layers 
        sequence_output = encoded_layers[-1]     #get the last layer output
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return embedding_output, encoded_layers, pooled_output

class TextLineForS2SDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining"""
    def __init__(self, config):
        super(TextLineForS2SDecoder, self).__init__(config)
        self.bert = TextLineModelIncr(config)      #this
        # note: the max source length is the max src seq length during fine tuning which includes the cls and sep
        # NOTE: we don't remove anything. the 0 is for padding
        self.cls = TextLineSPPreTrainingHeads(config, src_len=config.max_source_length, num_labels=2)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(self, input_ids, sentence_num,input_label_data, mask_qkv=None,WITH_LABEL=False):
        output_ids = []
        position_ids_all=[]
        input_mask_all=[]
        prev_embedding = None
        prev_encoded_layers = None
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        
        curr_ids = input_ids
        mask_ids = input_ids.new_zeros(batch_size, 1, input_ids.size(-1))
        if WITH_LABEL:
            curr_label=input_label_data
            mask_label=input_label_data.new_zeros(batch_size,1)
            mask_label[:,0]=6
        """ token_type 1 1024"""
        token_type_ids=torch.zeros(1024).to(dtype=torch.int64)
        token_type_ids[513:]=1
        token_type_ids = torch.stack([token_type_ids] * batch_size, dim=0)
        token_type_ids=token_type_ids.to(input_ids.device)
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]
        for i in range(batch_size):
            """ position 1 1024"""
            position_ids=torch.arange(0,sentence_num[i])
            pad=torch.zeros(513-sentence_num[i])
            position_ids_=torch.arange(sentence_num[i],sentence_num[i]+511)
            position_ids=torch.cat([position_ids,pad,position_ids_],dim=0)
            position_ids=position_ids.to(dtype=torch.int64)
            position_ids_all.append(position_ids)
            position_ids = torch.stack(position_ids_all, dim=0).to(input_ids.device)
            """"input mask 1024 1024 trill"""
            _tril_matrix = torch.tril(torch.ones((1024, 1024), dtype=torch.long))
            input_mask = torch.zeros(1024, 1024, dtype=torch.long) #input4=1024*1024
            input_mask[:, :sentence_num[i]].fill_(1)  #tringle mask
            input_mask[513:1024, 513:1024].copy_(_tril_matrix[:511, :511]) #input 4
            input_mask_all.append(input_mask)
        input_mask = torch.stack(input_mask_all, dim=0).to(input_ids.device)
        next_pos = input_length
        src_embedding = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            if WITH_LABEL:
                x_input_label=torch.cat((curr_label,mask_label),dim=1)
            else:
                x_input_label=None
            """current input n<1024"""
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = input_mask[:,start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            
            # curr_position_ids = position_ids[:, :]
            # curr_token_type_ids = token_type_ids[:,:]
            # curr_attention_mask = input_mask[:,:]
            # pad_length = 1024 - x_input_ids.size(1)
            # x_input_ids_ = F.pad(x_input_ids, (0, 0, pad_length//2, pad_length//2 + pad_length%2, 0, 0))
            
            new_embedding, new_encoded_layers, _ = self.bert(
                            x_input_ids, curr_token_type_ids, curr_position_ids, 
                            curr_attention_mask,input_label_data=x_input_label,      
                            output_all_encoded_layers=True, prev_embedding=prev_embedding,
                            prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv)            #bert

            if src_embedding is None:
                src_embedding = new_embedding[:, :-1, :]
                #src_embedding = new_embedding[:, 513: :]
                
            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores, _ = self.cls(last_hidden, None, src_embedding)  #cls 1 1 n,n<1024
            """getpredict"""
            _, max_ids = torch.max(prediction_scores, dim=-1)
            output_ids.append(max_ids-1)

            if prev_embedding is None:                       #this
                prev_embedding = new_embedding[:, :-1, :]
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
            if prev_encoded_layers is None:
                prev_encoded_layers = [x[:, :-1, :]
                                        for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                        for x in zip(prev_encoded_layers, new_encoded_layers)]
            _, _, dim = input_ids.shape
            if WITH_LABEL:
                curr_label=torch.gather(input_label_data,1,max_ids)
            index = max_ids.unsqueeze(-1)
            index = index.expand(index.shape[0], index.shape[1], dim)
            curr_ids = torch.gather(input_ids, 1, index)
            next_pos += 1
        out_put_ids=torch.cat(output_ids, dim=1)
        return out_put_ids
