from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from nss.generate import dino_scoring


def test_embed_image_l2_normalizes_pooler_output(tmp_path: Path) -> None:
    """embed_image L2-normalizes whatever `pooler_output` the (mocked) DINOv2 model returns.

    No real model weights loaded -- `_load_dino` is patched to return a fake model/processor pair,
    so this exercises embed_image's own normalization logic in isolation.
    """
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)

    fake_processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 8, 8)})
    fake_output = MagicMock(pooler_output=torch.tensor([[3.0, 4.0]]))  # norm = 5
    fake_model = MagicMock(return_value=fake_output)

    with patch.object(dino_scoring, "_load_dino", return_value=(fake_model, fake_processor)):
        result = dino_scoring.embed_image(image_path)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [0.6, 0.8], atol=1e-6)
    assert np.linalg.norm(result) == pytest.approx(1.0)
