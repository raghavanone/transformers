# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the PyTorch ConvNext model. """

import inspect
import unittest

from torch import nn

from transformers import ConvNextConfig, CrossformerConfig
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin

if is_torch_available():
    import torch

    from transformers import ConvNextBackbone, ConvNextForImageClassification, ConvNextModel
    from transformers.models.crossformer import CrossFormerModel, CrossFormerForImageClassification

if is_vision_available():
    from PIL import Image

    from transformers import AutoFeatureExtractor


class CrossformerModelTester:
    def __init__(
            self,
            parent,
            batch_size=1,
            img_size=224,
            patch_size=[4],
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            group_size=[7, 7, 7, 7],
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            activation_function="gelu",
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            merge_size=[[2], [2], [2]],
            use_labels=True,
            is_training=True,
            scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.group_size = group_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.activation_function = activation_function
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.ape = ape
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint
        self.merge_size = merge_size
        self.use_labels = use_labels

        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.in_chans, self.img_size, self.img_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_classes)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return CrossformerConfig(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            group_size=self.group_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            activation_function=self.activation_function,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            ape=self.ape,
            patch_norm=self.patch_norm,
            use_checkpoint=self.use_checkpoint,
            merge_size=self.merge_size,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = ConvNextModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expected last hidden states: B, C, H // 32, W // 32
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        model = ConvNextForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = ConvNextBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 4, 4])

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])

        # verify backbone works with out_features=None
        config.out_features = None
        model = ConvNextBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[-1], 1, 1])

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class CrossformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ConvNext does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            CrossFormerModel,
            CrossFormerForImageClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": CrossFormerModel, "image-classification": CrossFormerForImageClassification}
        if is_torch_available()
        else {}
    )

    # fx_compatible = True
    # test_pruning = False
    # test_resize_embeddings = False
    # test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = CrossformerModelTester(self)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        _ , inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                self.model_tester.batch_size, dtype=torch.long, device=torch_device
            )
        return inputs_dict

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            self.assertListEqual(arg_names[:1], ['pixel_values'])

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            if model_class.__name__ == "CrossFormerModel":
                return
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
                *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
            ]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()