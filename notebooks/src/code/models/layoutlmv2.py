# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""MLM implementations for LayoutLMv2/XLM

At the time of writing, Hugging Face (transformers v4.18) doesn't include masked language modelling
heads for LayoutLMv2/XLM - only v1... So here's a basic custom implementation inspired by the v1
head to support continuation pre-training.

Note that the canonical pre-training for these models includes additional objectives which this
setup doesn't capture (extra tasks to promote visual-text feature alignment).
"""

# Python Built-Ins:
from typing import Optional, Tuple, Union

# External Dependencies:
import torch
from torch import nn
from transformers import LayoutLMv2Model, LayoutLMv2PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMOnlyMLMHead


class LayoutLMv2ForMaskedLM(LayoutLMv2PreTrainedModel):
    """LayoutLMv2 with a custom MLM head inspired by transformers.LayoutLMForMaskedLM"""

    def __init__(self, config):
        super().__init__(config)

        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = LayoutLMOnlyMLMHead(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.classifier.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.classifier.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        prediction_scores = self.classifier(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
