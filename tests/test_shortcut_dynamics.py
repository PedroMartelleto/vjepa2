# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import torch.nn as nn
from src.models.shortcut_dynamics import GeometricDynamicsModel

class TestShortcutDynamicsModel(unittest.TestCase):

    def setUp(self):
        # Configuration mock
        self.img_size = 224
        self.num_frames = 8
        self.patch_size = 16
        self.tubelet_size = 2
        
        self.encoder_config = {
            "model_name": "vit_small", # Use small for faster testing
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "checkpoint_key": "target_encoder"
        }
        
        self.dynamics_config = {
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "action_dim": 7,
            "use_rope": False,
            "use_swiglu": False
        }
        
        self.decoder_config = {
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "num_point_queries": 5,
            "output_dim": 3
        }

        self.model = GeometricDynamicsModel(
            encoder_config=self.encoder_config,
            dynamics_config=self.dynamics_config,
            decoder_config=self.decoder_config,
            img_size=self.img_size,
            num_frames=self.num_frames
        )

    def test_forward_pass_shape(self):
        B = 2
        C = 3
        T_video = self.num_frames
        H = self.img_size
        W = self.img_size
        
        T_act = 10
        D_act = 7
        
        video = torch.randn(B, C, T_video, H, W)
        actions = torch.randn(B, T_act, D_act)
        
        output = self.model(video, actions)
        
        # Expected output shape: (B, num_point_queries, output_dim)
        expected_shape = (B, self.decoder_config["num_point_queries"], self.decoder_config["output_dim"])
        self.assertEqual(output.shape, expected_shape)

    def test_gradient_flow(self):
        B = 1
        video = torch.randn(B, 3, self.num_frames, self.img_size, self.img_size)
        actions = torch.randn(B, 5, 7)
        
        output = self.model(video, actions)
        loss = output.mean()
        loss.backward()
        
        # Check Encoder Gradients (Should be None or 0)
        for name, param in self.model.encoder.named_parameters():
            if param.requires_grad:
                self.assertFalse(param.requires_grad, f"Encoder param {name} should not require grad")
            # Even if requires_grad is False, .grad might be None
            self.assertIsNone(param.grad)

        # Check Dynamics Gradients (Should be non-zero)
        dynamics_has_grad = False
        for param in self.model.dynamics.parameters():
            if param.grad is not None and param.grad.sum() != 0:
                dynamics_has_grad = True
                break
        self.assertTrue(dynamics_has_grad, "Dynamics module should receive gradients")

        # Check Decoder Gradients
        decoder_has_grad = False
        for param in self.model.decoder.parameters():
            if param.grad is not None and param.grad.sum() != 0:
                decoder_has_grad = True
                break
        self.assertTrue(decoder_has_grad, "Decoder module should receive gradients")

    def test_overfitting(self):
        """Sanity check: can the model overfit a single sample?"""
        B = 1
        video = torch.randn(B, 3, self.num_frames, self.img_size, self.img_size)
        actions = torch.randn(B, 5, 7)
        target = torch.randn(B, self.decoder_config["num_point_queries"], 3)
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        criterion = nn.MSELoss()
        
        initial_loss = None
        for i in range(50):
            optimizer.zero_grad()
            output = self.model(video, actions)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss * 0.1, "Model failed to overfit on single batch")

    def test_pos_embedding_sensitivity(self):
        """Ensure different action order/positions produce different outputs (positional embeddings work)"""
        B = 1
        video = torch.randn(B, 3, self.num_frames, self.img_size, self.img_size)
        
        # Create two action sequences, one reversed
        T_act = 10
        actions = torch.randn(B, T_act, 7)
        actions_rev = actions.clone()
        actions_rev = torch.flip(actions_rev, [1])
        
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(video, actions)
            out2 = self.model(video, actions_rev)
            
        # Outputs should be different
        diff = (out1 - out2).abs().mean().item()
        self.assertGreater(diff, 1e-5, "Positional embeddings might not be working; reversed input gave same output")

if __name__ == '__main__':
    unittest.main()