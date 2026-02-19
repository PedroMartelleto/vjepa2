# Model Implementation Plan: Shortcut Dynamics

## 1. Architecture Overview

This model acts as a "residual" predictor that leverages a powerful, frozen visual backbone to understand the scene, while learning a lightweight dynamics module to predict the effects of actions.

### High-Level Data Flow

1.  **Video Input**: $V \in \mathbb{R}^{B \times C \times T \times H \times W}$
2.  **Encoder (Frozen V-JEPA)**:
    *  requires_grad(False)
    *   **Output**: Final feature map with the tokens encoded by JEPA (before pooling).
    *  JEPA tokens = mean-pooled spatial tokens per frame + one global token
3.  **ShortcutDynamicsModule**:
    *   **Input**: Action sequence $A \in \mathbb{R}^{B \times T_{act} \times D_{act}}$.
    *   **Processing**:
        1.  Project Actions + Add Absolute Positional Embeddings.
        2.  **Semantic Interaction**: Cross-Attend to the JEPA tokens.
    *   **Output**: Dynamics Tokens $D_{tokens} \in \mathbb{R}^{B \times T_{act} \times D}$.
4.  **PointDecoder**:
    *   **Input**: Query tokens (a torch vector).
    *   **Processing**: Cross-Attend to $D_{tokens}$.
    *   **Output**: 3D Coordinates $\in \mathbb{R}^{B \times N_{points} \times 3}$.

## 3. Configuration

We will add a new configuration section compatible with the existing YAML structure.

```yaml
model:
  model_name: geometric_dynamics_model
  encoder:
    model_name: vit_giant_xformers
    patch_size: 16
    tubelet_size: 2
    checkpoint_key: target_encoder
  dynamics:
    embed_dim: 1024 # Project actions to this dim
    depth: 4
    num_heads: 16
    mlp_ratio: 4.0
    action_dim: 7 
    use_rope: false 
    use_swiglu: true
  decoder:
    embed_dim: 512
    depth: 3
    num_heads: 8
    num_point_queries: 1
    output_dim: 3
```

## 4. Implementation Plan

We will implement a new file `src/models/shortcut_dynamics.py`. We will heavily reuse `src/models/utils/modules.py`.

### 4.1. Classes to Implement

#### 1. `ShortcutDynamicsModule` (extends `nn.Module`)
A lightweight Transformer that processes action tokens.

*   **Components**:
    *   `action_proj`: `nn.Linear` ($D_{act} \to D_{model}$).
    *   `pos_embed`: `nn.Parameter` (shape $1 \times T_{max} \times D_{model}$).
    *   `blocks`: `nn.ModuleList` of `CrossAttentionBlock` (reused from `src.models.utils.modules`).
*   **Logic**:
    *   The module receives JEPA tokens.
    *   **Blocks** performs Cross-Attention where $Q=Actions, K/V=JEPA tokens$.
    *   The existing `CrossAttentionBlock` in `modules.py` is `CrossAttn -> Add -> MLP`. It does *not* contain Self-Attention. We will insert a standard `Block` (Self-Attn) between Cross-Attentions to allow temporal mixing of actions.

#### 2. `PointDecoder` (extends `nn.Module`)
Reads out trajectory points from the dynamics tokens.

*   **Reuse**: This is functionally identical to `src.models.attentive_pooler.AttentiveClassifier`, but regression-based instead of classification.
*   **Components**:
    *   `query_tokens`: `nn.Parameter`.
    *   `blocks`: Stack of `CrossAttentionBlock`.
    *   `head`: `nn.Linear` ($D_{model} \to 2$).

#### 3. `GeometricDynamicsModel` (extends `nn.Module`)
The wrapper class to be called by the training loop.

*   **Init**:
    *   Initializes `src.models.vision_transformer.VisionTransformer` (frozen).
    *   Initializes `ShortcutDynamicsModule`.
    *   Initializes `PointDecoder`.
*   **Forward**:
    *   Extracts JEPA tokens from encoder (using `torch.no_grad()`).
    *   Passes actions + features to Dynamics.
    *   Passes dynamics output to Decoder.

### 4.2. Existing Modules to Reuse
*   `src.models.utils.modules.CrossAttentionBlock`: For Dynamics (Action $\leftrightarrow$ Image) and Decoder (Query $\leftrightarrow$ Dynamics).
*   `src.models.utils.modules.Block`: For Self-Attention within the Dynamics module (Action $\leftrightarrow$ Action).
*   `src.models.utils.modules.SwiGLUFFN`: For MLP layers (set `act_layer=nn.SiLU, wide_silu=True`).
*   `src.models.vision_transformer.VisionTransformer`: As the backbone.

## 5. Validations & Sanity Checks

Before launching full training, we will implement `tests/test_shortcut_model.py`:

1.  **Gradient Check**:
    *   Run a forward/backward pass.
    *   Assert `model.encoder.parameters().grad` is `None` or all zeros (Frozen).
    *   Assert `model.dynamics.parameters().grad` is non-zero (Trainable).

2.  **Shape Verification**:
    *   Input: $(B, C, 16, 256, 256)$.
    *   Encoder Output: Ensure list of 2 tensors returned with expected patch counts.
    *   Dynamics Output: Ensure shape is $(B, T_{action}, D)$.
    *   Decoder Output: Ensure shape is $(B, N_{points}, 2)$.

3.  **Overfitting Test**:
    *   Take a single batch of data (1 video, 1 action sequence, 1 ground truth point).
    *   Train for 100 steps.
    *   Assert Loss $\to 0$. This confirms the pipeline mechanics are correct.

4.  **Positional Embedding Check**:
    *   Pass actions with `T=10`. Shuffle the actions and pass them again. Ensure the output changes.