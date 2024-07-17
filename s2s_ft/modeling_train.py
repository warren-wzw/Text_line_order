from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import os
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from transformers.modeling_bert import \
    BertPreTrainedModel, BertSelfOutput, BertIntermediate, BertOutput, BertPredictionHeadTransform
from s2s_ft.config import BertForSeq2SeqConfig
logger = logging.getLogger(__name__)
BertLayerNorm = torch.nn.LayerNorm

class BertPreTrainedForSeq2SeqModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertForSeq2SeqConfig

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
class TextLineEmbeddings(nn.Module):
    def __init__(self, config):
        super(TextLineEmbeddings, self).__init__()
        """ embedding """
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        self.label_type_embeddings = nn.Embedding(10, config.hidden_size)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,bbox,input_label_data=None,input_sentence_len=None,token_type_ids=None,position_ids=None,inputs_embeds=None,):
        """position"""
        position_embeddings = self.position_embeddings(position_ids)
        if bbox.size(2)<8:
            """layout"""
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
            h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
            w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
            embeddings = (
                    left_position_embeddings
                    + upper_position_embeddings
                    + right_position_embeddings
                    + lower_position_embeddings
                    + h_position_embeddings
                    + w_position_embeddings
                    + position_embeddings) #n 1535 768
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
                    + position_embeddings) #n 1535 768
        if input_label_data is not None:
            label_embeddings=self.label_type_embeddings(input_label_data)
            embeddings=embeddings+label_embeddings
        """token type"""
        embeddings = embeddings + self.token_type_embeddings(token_type_ids) 

        embeddings = self.LayerNorm(embeddings.to(dtype=torch.float32))
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config): 
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs) #random set 0
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, split_lengths=None):
        mixed_query_layer = self.query(hidden_states)
        if split_lengths:
            assert not self.output_attentions

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states) #skip
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        if split_lengths:
            query_parts = torch.split(mixed_query_layer, split_lengths, dim=1)
            key_parts = torch.split(mixed_key_layer, split_lengths, dim=1)
            value_parts = torch.split(mixed_value_layer, split_lengths, dim=1)

            key = None
            value = None
            outputs = []
            sum_length = 0
            for (query, _key, _value, part_length) in zip(query_parts, key_parts, value_parts, split_lengths):
                key = _key if key is None else torch.cat((key, _key), dim=1)
                value = _value if value is None else torch.cat((value, _value), dim=1)
                sum_length += part_length
                outputs.append(self.multi_head_attention(query, key, value, 
                    attention_mask[:, :, sum_length - part_length: sum_length, :sum_length])[0])
            outputs = (torch.cat(outputs, dim=1), )
        else:
            outputs = self.multi_head_attention(
                mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask)#skip
        return outputs

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, split_lengths=None):
        self_outputs = self.self(hidden_states, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states, split_lengths=split_lengths)#1 1535 hidden
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # resnet 1 1535 hidden
        return outputs

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, split_lengths=None):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, split_lengths=split_lengths)
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output) #fc
        layer_output = self.output(intermediate_output, attention_output) 
        outputs = (layer_output,) + self_attention_outputs[1:] #fc+res
        return outputs
  
class BertEncoder_self(nn.Module):
    def __init__(self, config):
        super(BertEncoder_self, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, split_lengths=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,) #skip

            layer_outputs = layer_module(hidden_states, attention_mask, split_lengths=split_lengths) #bert layer
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) #skip

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,) #skip
        if self.output_attentions:
            outputs = outputs + (all_attentions,) #skip
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class TextLineModel(BertPreTrainedForSeq2SeqModel):
    def __init__(self, config):
        super(TextLineModel, self).__init__(config)
        self.config = config
        self.embeddings = TextLineEmbeddings(config)
        self.encoder = BertEncoder_self(config)

    def forward(self,
                bbox=None,
                input_label_data=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                split_lengths=None,
                input_sentence_len=None,
                return_emb=False):

        device = bbox.device
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :] #1 1 1535 1535

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :] #skip
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            bbox,input_label_data=input_label_data, position_ids=position_ids,input_sentence_len=input_sentence_len, 
            token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask, split_lengths=split_lengths)
        sequence_output = encoder_outputs[0] #1 1535 hidden

        outputs = (sequence_output, ) + encoder_outputs[1:]  # only last hiddenstate so encoder_outputs[1:]=()  

        if return_emb:
            outputs += (embedding_output,) #return model output and embedding output

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing=0, tgt_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_size > 0

        smoothing_value = label_smoothing / (tgt_size - 2)
        one_hot = torch.full((tgt_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_size = tgt_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1) #batch #511
        output = output.view(-1, self.tgt_size)
        target = target.view(-1) #511
        model_prob = self.one_hot.float().repeat(target.size(0), 1) #511，513 generat prob tensor
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)#use confidence fill the prob tensopr 
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0) #511 513 

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2) #计算模型预测概率分布与目标概率分布之间的 KL 散度,在进行差异度汇总

class CrossEntropyLoss(_Loss):
    def __init__(self, label_smoothing=0, tgt_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return self.criterion(output, target)

class LabelSmoothingLossMSE(_Loss):
    def __init__(self,):
        super(LabelSmoothingLossMSE, self).__init__()
    def forward(self, output, target):
        return F.mse_loss(output,target)

class TextLineSPLMPredictionHead(nn.Module):
    def __init__(self, config, src_len):
        super(TextLineSPLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.bias = nn.Parameter(torch.zeros(src_len))

    def forward(self, hidden_states, src_emb):
        """hidden state 1 511 192 src emb 1 513 192"""
        hidden_states = self.transform(hidden_states)
        hidden_states = torch.einsum('btf,bsf->bts', hidden_states, src_emb) + self.bias  #get hidden state
        return hidden_states

class TextLineSPLMPredictionHeadWithSelfAttention(nn.Module):
    def __init__(self, config, src_len):
        super(TextLineSPLMPredictionHeadWithSelfAttention, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.self_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)  # 自注意力机制
        self.bias = nn.Parameter(torch.zeros(src_len))
    def forward(self, hidden_states, src_emb):
        """hidden state 1 511 192 src emb 1 513 192"""
        hidden_states = self.transform(hidden_states)
        # 将 hidden_states 作为输入，计算自注意力
        # hidden_states: [seq_len, batch_size, hidden_size]
        # 注意：这里要把 batch_size 放在第二维，因为 PyTorch 中自注意力的输入格式要求为 [seq_len, batch_size, hidden_size]
        src_emb = src_emb.transpose(0, 1).contiguous()
        src_emb, _ = self.self_attention(src_emb, src_emb, src_emb)   
        src_emb = src_emb.transpose(0, 1).contiguous()
        hidden_states = torch.einsum('btf,bsf->bts', hidden_states, src_emb) + self.bias  # get hidden state
        return hidden_states

class TextLineHeadOnlyMLMHead(nn.Module):
    def __init__(self, config, src_len):
        super(TextLineHeadOnlyMLMHead, self).__init__()
        self.predictions = TextLineSPLMPredictionHeadWithSelfAttention(config, src_len=src_len)

    def forward(self, sequence_output, src_emb):
        prediction_scores = self.predictions(sequence_output, src_emb=src_emb)
        return prediction_scores

class TextLineOrderS2S(BertPreTrainedForSeq2SeqModel):
    def __init__(self, config):
        super(TextLineOrderS2S, self).__init__(config)
        self.bert = TextLineModel(config)
        self.cls = TextLineHeadOnlyMLMHead(config, src_len=config.max_source_length)
        self.init_weights()
        self.log_softmax = nn.LogSoftmax()
        self.source_type_id = config.source_type_id
        self.target_type_id = config.target_type_id
        self.max_source_length=config.max_source_length
        self.max_target_length=config.max_source_length-2
        """loss"""
        self.crit_mask_lm_smoothed=LabelSmoothingLoss(
                config.label_smoothing, config.max_source_length, ignore_index=0, reduction='none')
        self.crit_mask_lm_smoothed_cro=CrossEntropyLoss(
                config.label_smoothing, config.max_source_length, ignore_index=0, reduction='none')
        self.crit_mask_lm_smoothed_mse=LabelSmoothingLossMSE()

    @staticmethod
    def create_mask_and_position_ids(num_tokens, max_len, offset=None):
        base_position_matrix = torch.arange(
            0, max_len, dtype=num_tokens.dtype, device=num_tokens.device).view(1, -1)
        mask = (base_position_matrix < num_tokens.view(-1, 1)).type_as(num_tokens)
        if offset is not None:
            base_position_matrix = base_position_matrix + offset.view(-1, 1)
        position_ids = base_position_matrix * mask
        return mask, position_ids
    @staticmethod
    def create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids):
        """source mask 0-n:1...,n-513:0... target mask 0-n:1...,n-511:0..."""
        """source_position_ids 0...m,0... 513 target_span_ids m...2m,0..511"""
        weight = torch.cat((torch.zeros_like(source_position_ids), target_span_ids, -target_span_ids), dim=1)
        from_weight = weight.unsqueeze(-1)#1 1535 1
        to_weight = weight.unsqueeze(1) #1 1 1535

        true_tokens = (0 <= to_weight) & (torch.cat((source_mask, target_mask, target_mask), dim=1) == 1).unsqueeze(1) #b,1,1535
        true_tokens_mask = (from_weight >= 0) & true_tokens & (to_weight <= from_weight)
        pseudo_tokens_mask = (from_weight < 0) & true_tokens & (-to_weight > from_weight)
        pseudo_tokens_mask = pseudo_tokens_mask | ((from_weight < 0) & (to_weight == from_weight))

        return (true_tokens_mask | pseudo_tokens_mask).type_as(source_mask)

    def forward(self,input_sentence_coord,sentence_num,target_index,
                input_label_data=None,input_sentence_len=None):
        source_len = self.max_source_length #513
        target_len = self.max_target_length #511
        pseudo_len = self.max_target_length #511
        assert target_len == pseudo_len
        assert source_len > 0 and target_len > 0
        split_lengths = (source_len, target_len, pseudo_len)
        input_xys = input_sentence_coord
        token_type_ids = torch.cat(
              (torch.ones(input_sentence_coord.size(0),513) * self.source_type_id, #0
              torch.ones(input_sentence_coord.size(0),511) * self.target_type_id,#1
              torch.ones(input_sentence_coord.size(0),511) * self.target_type_id), dim=1) #1
        token_type_ids = token_type_ids.to(torch.long).to(input_sentence_coord.device)#0...,1....,1.....
        source_mask, source_position_ids = self.create_mask_and_position_ids(sentence_num+1, source_len)
        target_mask, target_position_ids = self.create_mask_and_position_ids(sentence_num, target_len, offset=sentence_num+1)
        position_ids = torch.cat((source_position_ids, target_position_ids, target_position_ids), dim=1)
        """0-512 1 513-1023 trill,1024-1535"""
        attention_mask = self.create_attention_mask(source_mask, target_mask, 
                                                    source_position_ids, target_position_ids) 
        """bert"""
        outputs = self.bert(input_xys, input_label_data=input_label_data,input_sentence_len=input_sentence_len,attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, split_lengths=split_lengths, return_emb=True) #this
        
        sequence_output = outputs[0]  #last hidden batch 1535 768z``
        sequence_embedding = outputs[-1]#sentence Advanced Representation
        pseudo_sequence_output = sequence_output[:, source_len + target_len:, ] #1,511,768
        source_embedding = sequence_embedding[:, :source_len, :] #1,513,768
        def caculate_accuarcy(prediction_scores_masked,sentence_num,target_index):
            log_probs = F.log_softmax(prediction_scores_masked.float(), dim=-1)
            predicted_labels = torch.argmax(log_probs, dim=-1)
            Accuarcy_=0.0
            target_index+=1    #test
            for i, row_len in enumerate(sentence_num):
                comparison_result = torch.eq(predicted_labels[i, :row_len], target_index[i, :row_len]).float().mean()
                Accuarcy_+=comparison_result.item()
            Accuarcy=Accuarcy_/(sentence_num.size(0)+1e-5)
            return Accuarcy
        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask                      # 将ignore mask part
            loss=torch.sum(loss)
            target_num = torch.sum(mask) + 1e-5    #1e-5防止分母为0
            return (loss / target_num).sum()       #计算掩码上的平均损失 
        def ChangeSeq(predicted_labels,target_index):
            predicted_seq_=[]
            for i in range(target_index.size(0)):
                seq_predict=[]
                seq_dict={}
                for index,seq_ in enumerate(target_index[i][:sentence_num[i].item()-1].tolist()):
                    seq_dict[seq_]=index
                for index,seq_ in enumerate(predicted_labels[i].tolist()):
                    if seq_ in seq_dict:
                        seq_predict.append(seq_dict[seq_])
                    else:
                        seq_predict.append(0)
                predicted_seq_.append(torch.tensor(seq_predict).to(target_index.device))
            predicted_seq = torch.stack(predicted_seq_, dim=0).float().to(target_index.device) 
            return predicted_seq
        """decode """
        prediction_scores_masked = self.cls(pseudo_sequence_output, source_embedding) #caculate scores 1,511,513
        Accuarcy=caculate_accuarcy(prediction_scores_masked,sentence_num,target_index)
        """loss"""
        tgt_index_ = [torch.arange(511) for _ in range(target_index.size(0))]
        tgt_index = torch.stack(tgt_index_).float().to(target_index.device)
        log_probs = F.log_softmax(prediction_scores_masked.float(), dim=-1)
        predicted_labels = torch.argmax(log_probs, dim=-1)
        predicted_seq=ChangeSeq(predicted_labels,target_index)
        masked_lm_loss_mse = (self.crit_mask_lm_smoothed_mse(tgt_index*target_mask,predicted_seq*target_mask)).float().requires_grad_(True)
        pseudo_lm_loss_mse = loss_mask_and_normalize(masked_lm_loss_mse.float(), target_mask) #cal loss and ignore mask loss
        """loss"""
        p_coord=F.log_softmax(prediction_scores_masked.float(),dim=-1) #softmax
        masked_lm_loss = self.crit_mask_lm_smoothed(p_coord, target_index) #the kl value between ori and predict 1 511

        pseudo_lm_loss = loss_mask_and_normalize(masked_lm_loss.float(), target_mask) #cal loss and ignore mask loss

        return pseudo_lm_loss+pseudo_lm_loss_mse,Accuarcy 