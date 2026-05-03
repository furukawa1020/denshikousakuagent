"""Deep learning models for MakerGraph AI.

The first production target is Maker Intent Encoder: a Transformer model
trained with contrastive learning so vague maker intent and feasible project
graphs live in the same embedding space.
"""

from .config import MakerIntentConfig

__all__ = ["MakerIntentConfig"]
