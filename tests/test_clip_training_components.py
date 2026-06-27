import unittest

from PIL import Image
import torch

from t2c_clip.clip_backbone import (
    PromptTextEncoder,
    TransformersCLIPImageEncoder,
    clip_projection_dim,
)
from t2c_clip.transforms import CLIPImageTransform


class CLIPTrainingComponentsTest(unittest.TestCase):
    def test_clip_image_transform_returns_first_pixel_values_row(self):
        processor = FakeImageProcessor()
        transform = CLIPImageTransform(processor)

        output = transform(Image.new("RGB", (2, 2)))

        self.assertTrue(torch.equal(output, torch.ones(3, 2, 2)))
        self.assertEqual(processor.call_count, 1)

    def test_transformers_clip_image_encoder_calls_get_image_features(self):
        clip = FakeCLIP(projection_dim=2)
        encoder = TransformersCLIPImageEncoder(clip)

        output = encoder(torch.ones(2, 3, 2, 2))

        self.assertTrue(torch.equal(output, torch.full((2, 2), 2.0)))
        self.assertTrue(clip.image_called)

    def test_prompt_text_encoder_projects_mean_prompt(self):
        encoder = PromptTextEncoder(prompt_embedding_dim=3, output_dim=2)
        with torch.no_grad():
            weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            encoder.projection.weight.copy_(weight)
            encoder.projection.bias.zero_()

        output = encoder(torch.tensor([[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]]]))

        self.assertTrue(torch.equal(output, torch.tensor([[2.0, 3.0]])))

    def test_clip_projection_dim_reads_positive_config_value(self):
        self.assertEqual(clip_projection_dim(FakeCLIP(projection_dim=7)), 7)


class FakeImageProcessor:
    def __init__(self):
        self.call_count = 0

    def __call__(self, images, return_tensors):
        self.call_count += 1
        self.return_tensors = return_tensors
        self.image_mode = images.mode
        return {"pixel_values": torch.ones(1, 3, 2, 2)}


class FakeCLIP(torch.nn.Module):
    def __init__(self, projection_dim: int):
        super().__init__()
        self.config = FakeConfig(projection_dim)
        self.image_called = False

    def get_image_features(self, pixel_values):
        self.image_called = True
        return torch.full((pixel_values.shape[0], self.config.projection_dim), 2.0)


class FakeConfig:
    def __init__(self, projection_dim: int):
        self.projection_dim = projection_dim
