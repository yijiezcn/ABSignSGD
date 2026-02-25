"""
ABSignSGD: Async Block SignSGD Optimizer

This is a clean, open-source implementation of the ABSignSGD algorithm
from the ICLR 2026 paper "Asynchronous Block-wise Optimization for
Memory-Efficient Neural Network Training".

The optimizer implements block-wise sign gradient descent where only one
transformer layer is updated per step, selected based on a depth-biased
scheduling scheme.

Paper Algorithm (Algorithm 1, Single Agent):
    for k = 0, 1, 2, ... do
        Select block i_k
        v ← sign(g_πik(x^k))
        x^k+1_πik ← x^k_πik - α·v
        x^k+1_πi ← x^k_πi for all i ≠ i_k
    end for

Supported Architectures:
    - LLaMA, Mistral (model.layers)
    - GPT models (transformer.h)
    - Pythia models (gpt_neox.layers)
    - T5-style models (decoder.layers)
    - RWKV models (blocks)
    - MPT models (transformer.blocks)
    - Bloom models (h)
    - OPT models (decoder.layers)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Literal
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer


def get_num_layers(model: torch.nn.Module) -> int:
    """
    Detect the number of transformer layers in a model.

    Supports various architectures including LLaMA, GPT, Pythia, T5, RWKV, etc.

    Args:
        model: A PyTorch neural network model

    Returns:
        Number of transformer layers

    Raises:
        ValueError: If transformer layers cannot be detected
    """
    layer_patterns = [
        'model.layers',       # LLaMA, Mistral, etc.
        'transformer.h',      # GPT models
        'gpt_neox.layers',    # Pythia models
        'h',                  # Some older models, Bloom
        'decoder.layers',     # T5-style, OPT
        'blocks',             # RWKV models
        'model.decoder.layers',  # Old facebook models
        'transformer.blocks', # MPT models
        'transformer.layers', # GPT-Neo models
    ]

    for pattern in layer_patterns:
        try:
            layers = model
            for part in pattern.split('.'):
                layers = getattr(layers, part)
            return len(layers)
        except AttributeError:
            continue

    raise ValueError(
        "Could not determine number of layers in model. "
        "ABSignSGD requires a model with recognizable transformer layer structure. "
        f"Supported patterns: {layer_patterns}"
    )


class ABSignSGD(Optimizer):
    """
    Async Block SignSGD Optimizer for Memory-Efficient Training.

    This optimizer implements the ABSignSGD algorithm which updates one
    transformer layer at a time using sign gradients. It is designed for
    memory-efficient training of large language models.

    Key Features:
        - Memory efficient: Only maintains optimizer state for one layer
        - FP16 training with gradient scaling (default)
        - Depth-biased block selection following paper's formula
        - Supports multiple transformer architectures

    Args:
        model (torch.nn.Module): Transformer model to optimize
        lr (float): Learning rate. Default: 1e-3
        depth_bias (float): Depth bias parameter 'c' in paper's formula
            τ_i = N + c(N - i). Default: 10.0 (paper's experimental value)
        selection_scheme (str): Block selection strategy. Either 'depth_biased'
            (paper's default) or 'uniform'. Default: 'depth_biased'
        fp16 (bool): Use FP16 training with gradient scaling. Default: True
        weight_decay (float): Weight decay coefficient. Applied BEFORE sign
            update. Default: 0.01
        gradient_clipping (float, optional): Gradient clipping threshold.
            Default: 0 (disabled). SignSGD handles large gradients naturally
            through the sign operation.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
        >>> optimizer = ABSignSGD(model, lr=1e-4)
        >>>
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     outputs = model(**batch)
        ...     loss = outputs.loss
        ...     loss.backward()
        ...     optimizer.step()

    Note:
        - FP16 mode uses GradScaler for numerical stability
        - Weight decay is applied before the sign update
        - Gradient clipping is disabled by default (SignSGD handles large gradients)

    References:
        Paper: "Asynchronous Block-wise Optimization for Memory-Efficient
               Neural Network Training" (ICLR 2026)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        depth_bias: float = 10.0,
        selection_scheme: Literal['depth_biased', 'uniform'] = 'depth_biased',
        fp16: bool = True,
        weight_decay: float = 0.01,
        gradient_clipping: Optional[float] = None,
    ):
        # Validate model has transformer layers
        self.model = model
        self.num_layers = self._validate_transformer_model()

        # Initialize Optimizer base class with dummy param group
        first_param = next(model.parameters())
        defaults = dict(lr=lr)
        super().__init__([{'params': [first_param]}], defaults)

        # Store configuration
        self.depth_bias = depth_bias
        self.selection_scheme = selection_scheme
        self.fp16 = fp16
        self.weight_decay = weight_decay

        # Set gradient clipping: default is 0 (disabled for SignSGD)
        # SignSGD handles large gradients naturally through sign operation
        if gradient_clipping is None:
            self.gradient_clipping = 0.0
        else:
            self.gradient_clipping = gradient_clipping

        # Initialize GradScaler for FP16
        self.scaler = GradScaler() if fp16 else None

        # State tracking
        self.selected_layer = None
        self.next_available_time = None  # For depth-biased selection

        # Cache layer parameter mappings
        self.layer_param_mappings = {
            i: self._get_layer_param_names(i)
            for i in range(self.num_layers)
        }

        # Initialize first block
        self._select_and_prepare_block()

    def _validate_transformer_model(self) -> int:
        """
        Validate that the model has recognizable transformer layers.

        Returns:
            Number of transformer layers

        Raises:
            ValueError: If model doesn't have recognizable transformer structure
        """
        try:
            num_layers = get_num_layers(self.model)
            if num_layers is None or num_layers <= 0:
                raise ValueError(
                    "Model does not appear to have transformer layers. "
                    "ABSignSGD is designed for transformer models only."
                )
            return num_layers
        except Exception as e:
            raise ValueError(
                f"Could not detect transformer layers in model: {e}\n"
                "ABSignSGD requires a model with recognizable transformer layer structure. "
                "Supported: LLaMA, GPT, Pythia, T5, RWKV, MPT, Bloom, OPT"
            ) from e

    def _get_layer_param_names(self, layer_idx: int) -> List[str]:
        """
        Get parameter names for a specific transformer layer.

        Supports multiple architectures including LLaMA, GPT, Pythia, T5, etc.

        Args:
            layer_idx: Index of the layer

        Returns:
            List of parameter names belonging to the layer

        Raises:
            ValueError: If no parameters found for the layer
        """
        # Define patterns for different architectures
        layer_patterns = [
            f"model.layers.{layer_idx}.",        # LLaMA, Mistral
            f"transformer.h.{layer_idx}.",       # GPT models
            f"gpt_neox.layers.{layer_idx}.",     # Pythia
            f"h.{layer_idx}.",                   # Bloom, older models
            f"decoder.layers.{layer_idx}.",      # T5-style, OPT
            f"blocks.{layer_idx}.",              # RWKV
            f"model.decoder.layers.{layer_idx}.", # Old facebook
            f"transformer.blocks.{layer_idx}.",  # MPT
            f"transformer.layers.{layer_idx}.",  # GPT-Neo
            f"layers.{layer_idx}.",              # Generic fallback
        ]

        param_names = []
        for name, _ in self.model.named_parameters():
            if any(pattern in name for pattern in layer_patterns):
                param_names.append(name)

        if not param_names:
            # Fallback: look for parameters containing the layer index
            fallback_params = [
                name for name, _ in self.model.named_parameters()
                if f".{layer_idx}." in name or f"[{layer_idx}]." in name
            ]
            if fallback_params:
                return fallback_params

            raise ValueError(
                f"No parameters found for layer {layer_idx}. "
                f"Model architecture may not be supported. "
                f"First 10 param names: {[n for n, _ in self.model.named_parameters()][:10]}"
            )

        return sorted(param_names)

    def _get_layer_parameters(self, layer_idx: int) -> List[nn.Parameter]:
        """
        Get parameter objects for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            List of Parameter objects for the layer
        """
        param_names = self.layer_param_mappings[layer_idx]
        return [p for n, p in self.model.named_parameters() if n in param_names]

    def _set_layer_grad(self, layer_idx: int):
        """
        Enable gradients only for a specific layer.

        Args:
            layer_idx: Index of the layer to enable gradients for
        """
        param_names = self.layer_param_mappings[layer_idx]

        for name, param in self.model.named_parameters():
            if name in param_names:
                param.requires_grad = True
            else:
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None

    def _select_block_depth_biased(self) -> int:
        """
        Select block using depth-biased scheduling from paper.

        Paper formula: τ_i = N + c(N - i + 1)
        where N is total layers, c is depth_bias (default 10), i is layer index.

        The layer with smallest τ_i (earliest available time) is selected.
        After selection, τ_i is incremented by the computation time.

        Returns:
            Index of the selected layer
        """
        if self.next_available_time is None:
            N = self.num_layers
            c = self.depth_bias
            # Initialize: τ_i = N + c(N - i)
            self.next_available_time = np.array([
                N + c * (N - i) for i in range(N)
            ], dtype=np.float64)

        # Select layer with minimum available time
        selected = int(np.argmin(self.next_available_time))

        # Update available time: add computation cost
        N = self.num_layers
        c = self.depth_bias
        self.next_available_time[selected] += N + c * (N - selected)

        return selected

    def _select_block_uniform(self) -> int:
        """
        Select block uniformly at random.

        Returns:
            Index of the randomly selected layer
        """
        return np.random.randint(0, self.num_layers)

    def _select_block(self) -> int:
        """
        Select a layer based on the configured selection scheme.

        Returns:
            Index of the selected layer
        """
        if self.selection_scheme == 'depth_biased':
            return self._select_block_depth_biased()
        elif self.selection_scheme == 'uniform':
            return self._select_block_uniform()
        else:
            raise ValueError(f"Unknown selection scheme: {self.selection_scheme}")

    def _select_and_prepare_block(self):
        """
        Select a layer and prepare it for updating.

        Only reinitializes if the layer has changed.
        """
        current_layer = self._select_block()

        if current_layer != self.selected_layer:
            # Enable gradients only for selected layer
            self._set_layer_grad(current_layer)
            self.selected_layer = current_layer

        # Update learning rate from scheduler (if any)
        # The param_groups[0]['lr'] may be updated by external scheduler

    def _apply_sign_update(self):
        """
        Apply the sign gradient update to the selected layer.

        Update rule: param = param - lr * sign(grad) - lr * weight_decay * param
        Weight decay is applied BEFORE sign update.
        """
        layer_params = self._get_layer_parameters(self.selected_layer)
        lr = self.param_groups[0]['lr']

        for p in layer_params:
            if p.grad is None:
                continue

            # Weight decay BEFORE sign update
            if self.weight_decay != 0:
                p.data.add_(-lr * self.weight_decay * p.data)

            # Sign update
            p.data.add_(-lr * torch.sign(p.grad))

    def _step_fp16(self):
        """
        Perform optimization step in FP16 with gradient scaling.

        Handles inf/nan detection and gradient clipping.
        Uses a dummy optimizer for GradScaler tracking.
        """
        # Check for inf/nan gradients
        found_inf = False
        found_nan = False

        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any():
                    found_inf = True
                if torch.isnan(param.grad).any():
                    found_nan = True
                if found_inf and found_nan:
                    break

        # Setup dummy optimizer for scaler tracking
        device = next(iter(self.model.parameters())).device
        dummy_param = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))
        optimizer_dummy = torch.optim.SGD([dummy_param], lr=0.1)

        if found_inf:
            dummy_param.grad = torch.tensor([float('inf')], device=device)
        else:
            dummy_param.grad = torch.tensor([1.0], device=device)

        self.scaler.unscale_(optimizer_dummy)

        if not found_nan:
            # Apply gradient clipping
            if self.gradient_clipping != 0:
                layer_params = self._get_layer_parameters(self.selected_layer)
                torch.nn.utils.clip_grad_norm_(
                    layer_params,
                    max_norm=self.gradient_clipping
                )

            # Apply sign update
            self._apply_sign_update()
        else:
            print("NaN gradients detected, skipping update")

        if found_inf:
            print(f"Inf gradients detected. Scale: {self.scaler._scale.item()}")

    def _step_fp32(self):
        """
        Perform optimization step in FP32 without gradient scaling.
        """
        # Apply gradient clipping
        if self.gradient_clipping != 0:
            layer_params = self._get_layer_parameters(self.selected_layer)
            torch.nn.utils.clip_grad_norm_(
                layer_params,
                max_norm=self.gradient_clipping
            )

        # Apply sign update
        self._apply_sign_update()

    def step(self):
        """
        Perform a single optimization step.

        This method applies the sign gradient update to the currently
        selected layer, then selects the next layer for updating.
        """
        if self.fp16:
            self._step_fp16()
        else:
            self._step_fp32()

        # Select next block
        self._select_and_prepare_block()

    def zero_grad(self, set_to_none: bool = True):
        """
        Reset gradients of the model.

        Args:
            set_to_none: Whether to set gradients to None instead of zero
        """
        for param in self.model.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
