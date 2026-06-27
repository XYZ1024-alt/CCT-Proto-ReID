import unittest

import torch

from t2c_clip.evaluation import evaluate_reid
from t2c_clip.model import T2CClipModel
from t2c_clip.prompts import PromptBank, PromptConfig


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class PromptMeanEncoder(torch.nn.Module):
    def forward(self, prompts):
        return prompts.mean(dim=1)


class EvaluationModelTest(unittest.TestCase):
    def test_evaluate_reid_excludes_same_camera_matches(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metrics = evaluate_reid(query, gallery, [1], [1, 2], [1], [1, 2], ranks=(1,))
        self.assertEqual(metrics.map, 0.0)
        self.assertEqual(metrics.cmc[1], 0.0)

    def test_evaluate_reid_reports_rank1_and_map(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metrics = evaluate_reid(query, gallery, [1], [2, 1], [1], [2, 2], ranks=(1,))
        self.assertEqual(metrics.map, 0.5)
        self.assertEqual(metrics.cmc[1], 0.0)

    def test_model_inference_uses_global_and_camera_prompts(self):
        prompt_bank = PromptBank(PromptConfig(num_cameras=1, num_train_ids=1, context_length=1, embedding_dim=2))
        with torch.no_grad():
            prompt_bank.global_prompt.zero_()
            prompt_bank.camera_prompts[0] = torch.tensor([[0.0, 1.0]])
            prompt_bank.identity_prompts[0] = torch.tensor([[10.0, 0.0]])
        model = T2CClipModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)

        output = model.encode_retrieval(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]])
        self.assertTrue(torch.allclose(output, expected))
