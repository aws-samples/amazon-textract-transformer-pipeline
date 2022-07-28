# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Pre-training (MLM) implementations for LayoutLMv2/XLM

At the time of writing, Hugging Face (transformers v4.18) doesn't include masked language modelling
heads for LayoutLMv2/XLM - only v1... So here's a custom implementation inspired by the existing
LayoutLMForMaskedLM, LayoutLMv2ForTokenClassification, and LayoutLMv2ForSequenceClassification
classes.

Canonical pre-training for LayoutLMv2 (per the paper) is multi-objective rather than just MLM, so
we create a setup along those lines but still output a MaskedLM-like output for consistency with
the Hugging Face 'mlm' task.
"""

# Python Built-Ins:
from typing import Optional, Tuple, Union

# External Dependencies:
import torch
from torch import nn
from transformers import LayoutLMv2Config, LayoutLMv2Model, LayoutLMv2PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMOnlyMLMHead


class LayoutLMv2ForPretraining(LayoutLMv2PreTrainedModel):
    """LayoutLMv2 with losses for multi-objective pre-training

    Core structure is inspired by v1 transformers.LayoutLMForMaskedLM, and output structure aligns
    to the core MLM ("MVLM" as described in LayoutLMv2 paper) objective. However, two additional
    heads are added for the Text-Image Matching (sequence classification) and Text-Image Alignment
    (token classification) pre-training objectives described in the paper. These losses are
    calculated and added to the returned "loss" tensor, when the corresponding "image_mask_label"
    and/or "imline_mask_label" inputs are provided.
    """

    def __init__(self, config: LayoutLMv2Config):
        super().__init__(config)

        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Standard Masked Visual-Language Modelling head:
        self.classifier = LayoutLMOnlyMLMHead(config)
        # Custom Text-Image Matching head (2 labels match, not-match)
        self.image_mask_classifier = nn.Linear(config.hidden_size * 3, 2)
        # Custom Text-Image Alignment head (2 labels masked, not-masked)
        self.imline_mask_classifier = nn.Linear(config.hidden_size, 2)

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
        image_mask_label: Optional[torch.Tensor] = None,
        imline_mask_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """Calculate forward pass for pre-training

        Note this implementation is designed for training only and not inference: TIA & TIM heads
        are only calculated if the corresponding `imline_mask_label` and `image_mask_label` labels
        are provided, and the returned MaskedLMOutput doesn't include their prediction logits (only
        the total `loss` including their contribution). You could change this if you wanted to use
        the model directly for TIM+TIA+MLM inference for some reason.
        """
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

        # Token cls tasks need the text part only; sequence cls tasks will use image embeds too:
        raw_sequence_output, final_image_embeddings = (
            outputs[0][:, :seq_length],
            outputs[0][:, seq_length:],
        )
        sequence_output = self.dropout(raw_sequence_output)

        # Main/core Masked Language Modelling output:
        mlm_prediction_scores = self.classifier(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                mlm_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        # Only calculate TIM loss if TIM labels are provided:
        if image_mask_label is not None:
            # Pooling / classifier input preparation here should be as per
            # LayoutLMv2ForSequenceClassification forward logic:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            visual_shape = list(input_shape)
            visual_shape[1] = (
                self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
            )
            visual_shape = torch.Size(visual_shape)
            final_shape = list(input_shape)
            final_shape[1] += visual_shape[1]
            final_shape = torch.Size(final_shape)
            visual_bbox = self.layoutlmv2._calc_visual_bbox(
                self.config.image_feature_pool_shape, bbox, device, final_shape
            )
            visual_position_ids = torch.arange(
                0, visual_shape[1], dtype=torch.long, device=device
            ).repeat(input_shape[0], 1)
            initial_image_embeddings = self.layoutlmv2._calc_img_embeddings(
                image=image,
                bbox=visual_bbox,
                position_ids=visual_position_ids,
            )

            # Average-pool the visual embeddings
            pooled_initial_image_embeddings = initial_image_embeddings.mean(dim=1)
            pooled_final_image_embeddings = final_image_embeddings.mean(dim=1)
            # Concatenate with the CLS token output and *then* apply the dropout:
            tim_sequence_output = self.dropout(
                torch.cat(
                    [
                        raw_sequence_output[:, 0, :],  # (CLS token final output)
                        pooled_initial_image_embeddings,
                        pooled_final_image_embeddings,
                    ],
                    dim=1,
                )
            )
            tim_logits = self.image_mask_classifier(tim_sequence_output)
            tim_loss_fct = nn.CrossEntropyLoss()
            tim_loss = tim_loss_fct(tim_logits.view(-1, 2), image_mask_label.view(-1))
        else:
            tim_loss = None

        # Only calculate TIA loss if TIA labels are provided:
        if imline_mask_label is not None:
            # As per LayoutLMv2ForTokenClassification logic (re-using same dropout as MLM):
            tia_logits = self.imline_mask_classifier(sequence_output)
            tia_loss_fct = nn.CrossEntropyLoss()
            tia_loss = tia_loss_fct(tia_logits.view(-1, 2), imline_mask_label.view(-1))
        else:
            tia_loss = None

        # Output is per the plain MLM, but with additional TIM and TIA losses added if present:
        # (Could output tia_logits, tim_logits if wanted)
        loss_components = [l for l in (masked_lm_loss, tim_loss, tia_loss) if l is not None]
        total_loss = sum(loss_components) if len(loss_components) else None

        if not return_dict:
            output = (mlm_prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MaskedLMOutput(
            loss=total_loss,
            logits=mlm_prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
