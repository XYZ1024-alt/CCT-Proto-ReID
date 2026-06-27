import unittest

import torch

from t2c_clip.losses import batch_hard_triplet_loss, bidirectional_contrastive_loss
from t2c_clip.tfc import TFCCenterBank


class TFCLossTest(unittest.TestCase):
    def test_tfc_updates_identity_centers_with_ema(self):
        bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        bank.update(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
        bank.update(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))

        expected = torch.tensor([2 ** -0.5, 2 ** -0.5])
        self.assertTrue(torch.allclose(bank.centers[0], expected, atol=1e-6))

    def test_tfc_loss_uses_existing_centers(self):
        bank = TFCCenterBank(num_train_ids=1, feature_dim=2, momentum=0.5)
        bank.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

        loss = bank.loss(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_bidirectional_contrastive_loss_is_low_for_matching_pairs(self):
        image = torch.eye(2)
        text = torch.eye(2)
        loss = bidirectional_contrastive_loss(image, text, logit_scale=10.0)
        self.assertLess(float(loss), 0.01)

    def test_batch_hard_triplet_loss_penalizes_close_negative(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        labels = torch.tensor([0, 0, 1])
        loss = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertGreater(float(loss), 0.0)
