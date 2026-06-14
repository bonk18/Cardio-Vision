"""
CardioVision — 1D Grad-CAM for ECG Explainability
Generates visual heatmaps highlighting which ECG regions drive predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping adapted for 1D signals.

    Hooks into the last convolutional layer of the CNN encoder to:
    1. Capture forward activations (feature maps)
    2. Capture backward gradients of the target class score
    3. Compute importance weights via global average pooling of gradients
    4. Generate a weighted activation heatmap over the input signal

    This reveals which temporal segments of the ECG waveform
    (P-wave, QRS complex, T-wave, ST segment) are most influential
    for the model's arrhythmia classification decision.
    """

    def __init__(self, model, target_layer=None):
        """
        Parameters
        ----------
        model : HybridCNNLSTM model instance.
        target_layer : nn.Module — the conv layer to hook.
                       If None, auto-detects the last Conv1d in the CNN encoder.
        """
        self.model = model
        self.model.eval()

        # Auto-detect target layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _find_last_conv(self):
        """Find the last Conv1d layer in the model's CNN encoder."""
        last_conv = None
        cnn = self.model.cnn if hasattr(self.model, 'cnn') else self.model
        for module in cnn.modules():
            if isinstance(module, torch.nn.Conv1d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv1d layer found in model.")
        return last_conv

    def _save_activation(self, module, input, output):
        """Forward hook: save feature map activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradients flowing back through the layer."""
        self.gradients = grad_output[0].detach()

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Compute the Grad-CAM heatmap for a single input.

        Parameters
        ----------
        input_tensor : (1, 1, seq_len) — single ECG beat.
        target_class : Class index to explain. If None, uses predicted class.

        Returns
        -------
        heatmap : (seq_len,) — normalized [0,1] importance heatmap.
        predicted_class : The class the model predicted.
        class_probs : (num_classes,) — softmax probabilities.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dim

        # Enable gradients for input
        input_tensor.requires_grad_(True)

        # Disable cuDNN so LSTM backward works in eval mode
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        # Forward pass
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1).detach()

        predicted_class = output.argmax(dim=1).item()
        if target_class is None:
            target_class = predicted_class

        # Backward pass for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Restore cuDNN setting
        torch.backends.cudnn.enabled = prev_cudnn

        # Compute Grad-CAM weights
        gradients = self.gradients[0]    # (C, L')
        activations = self.activations[0]  # (C, L')

        # Global average pooling of gradients → channel importance weights
        weights = gradients.mean(dim=1)  # (C,)

        # Weighted combination of feature maps
        cam = torch.zeros(activations.shape[1], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap to match input length
        input_len = input_tensor.shape[-1]
        if len(cam) != input_len:
            cam = np.interp(
                np.linspace(0, 1, input_len),
                np.linspace(0, 1, len(cam)),
                cam,
            )

        return cam, predicted_class, probs[0].detach().cpu().numpy()

    def compute_batch(
        self,
        inputs: torch.Tensor,
        target_classes: Optional[list] = None,
    ) -> list:
        """
        Compute Grad-CAM for a batch of inputs.

        Returns list of (heatmap, predicted_class, class_probs) tuples.
        """
        results = []
        for i in range(inputs.shape[0]):
            single = inputs[i:i+1]
            tc = target_classes[i] if target_classes else None
            results.append(self.compute(single, tc))
        return results

    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass
