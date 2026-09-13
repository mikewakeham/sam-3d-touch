"""Explicit diagnostic dropout; preserve historical train.py and its hashes."""
import torch
from train import TouchTrainingModel


class UpperBoundModel(TouchTrainingModel):
    def __init__(self, generator, encoder, oracle, constant_features=None):
        super().__init__(generator, encoder, False, oracle)
        # Raw frozen VecSetX features, never projected tokens/target geometry.
        self.register_buffer('constant_features', constant_features, persistent=False)

    def surface_tokens(self, points, mask):
        if self.touch_encoder is None:
            return None
        encoder = self.touch_encoder
        if self.constant_features is None:
            return encoder(points, mask)
        features = self.constant_features.expand(len(points), -1, -1)
        return encoder.output_projection(features) + encoder.touch_embedding

    def forward(self, targets, condition_args, condition_kwargs, points, mask,
                drop_visual=False):
        if len(condition_args) != 1 or condition_kwargs:
            raise ValueError('Expected the established single visual-token context')
        visual = condition_args[0]
        if drop_visual:
            visual = torch.zeros_like(visual)
        tokens = self.surface_tokens(points, mask)
        kwargs = {} if tokens is None else {'touch_tokens': tokens}
        loss, _ = self.generator.loss(targets, visual, **kwargs)
        return loss
