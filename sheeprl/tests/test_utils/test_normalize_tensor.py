import torch
from sheeprl.utils.utils import normalize_tensor


def test_normalize_tensor_unmasked():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = normalize_tensor(tensor)
    expected = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    assert result.shape == tensor.shape
    assert torch.allclose(result, expected)


def test_normalize_tensor_with_mask():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])
    result = normalize_tensor(tensor, mask=mask)
    masked = tensor[mask]
    expected_masked = (masked - masked.mean()) / (masked.std() + 1e-8)
    expected = torch.zeros_like(tensor)
    expected[mask] = expected_masked
    assert result.shape == tensor.shape
    assert torch.allclose(result, expected)
