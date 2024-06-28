import numpy as np
import timm
import torch
from vision_transformer import VisionTransformer


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_tensors_equal(tensor1, tensor2):
    array1, array2 = tensor1.detach().numpy(), tensor2.detach().numpy()
    np.testing.assert_allclose(array1, array2)

# Load the pre-trained model from timm
model_identifier = "vit_base_patch16_384"
pretrained_model = timm.create_model(model_identifier, pretrained=True)
pretrained_model.eval()
print(type(pretrained_model))

# Configuration for the custom Vision Transformer model
vision_transformer_config = {
    "img_size": 384,
    "in_chans": 3,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4,
}

# Initialize the custom Vision Transformer model
custom_model = VisionTransformer(**vision_transformer_config)
custom_model.eval()

# Transfer weights from the pre-trained model to the custom model
for (name_pretrained, param_pretrained), (name_custom, param_custom) in zip(
        pretrained_model.named_parameters(), custom_model.named_parameters()
):
    assert param_pretrained.numel() == param_custom.numel()
    print(f"{name_pretrained} | {name_custom}")

    param_custom.data[:] = param_pretrained.data

    verify_tensors_equal(param_custom.data, param_pretrained.data)

# Create a random input tensor and perform inference
input_tensor = torch.rand(1, 3, 384, 384)
output_custom = custom_model(input_tensor)
output_pretrained = pretrained_model(input_tensor)

# Save the custom model
torch.save(custom_model, "model.pth")
