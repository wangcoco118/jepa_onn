# V-JEPA 中 ONN Predictor 架构说明

## 一、整体总结

**当前新增的 ONN 路线不是在原始 Transformer Predictor 内部仅替换某一个 QKV 层，而是用一个新的 `ONNFeedbackPredictor` 整体替换原始电子 Transformer Predictor。V-JEPA 的上下文编码器和目标编码器仍然保留，二者从预训练 checkpoint 加载后冻结；新增 Predictor 接收上下文特征，将 1568 个时空 token 组织成 8 个连续的 196-token 数据块，依次送入同一个自由空间光学网络，并利用前一个数据块的输出形成跨块反馈。**

当前 IntPhys 配置使用的是公开的 V-JEPA ViT-L/16：编码器为 `vit_large`，checkpoint 为 `vitl16.pth.tar`。编码器输出维度为 1024，新增 Predictor 的内部维度为 384，ONN 输入和输出均为 384，最后通过一个电子 MLP 将 384 维结果恢复为 1024 维，与冻结的目标编码器特征进行比较。

当前新增路线的核心数据流可以概括为：

```text
输入视频 [B, 3, 16, 224, 224]
        │
        ├── 冻结的 context encoder
        │       └── 上下文 token [B, N_ctxt, 1024]
        │
        ├── 冻结的 target encoder
        │       └── 目标 token [B, N_tgt, 1024]
        │
        └── ONNFeedbackPredictor
                │
                ├── 1024 → 384
                ├── 按原始位置 [B, 1568, 384]
                ├── 划分为 8 个 [B, 196, 384] 数据块
                ├── 8 次串行复用同一个 FeedbackFSONN
                ├── 拼接回 [B, 1568, 384]
                ├── 抽取 target 位置
                └── 384 → MLP → 1024
        │
        └── 与 target encoder 的目标特征计算预测损失
```

当前配置中的关键参数如下：

| 模块               | 当前实现                                     |
| ------------------ | -------------------------------------------- |
| V-JEPA 编码器      | `vit_large`，ViT-L/16                      |
| 预训练 checkpoint  | `vitl16.pth.tar`                           |
| 视频输入           | 16 帧，224×224，tubelet size=2              |
| 时空 token 数      | 8×14×14 = 1568                             |
| Predictor 输入维度 | 1024                                         |
| Predictor 内部维度 | 384                                          |
| ONN 输入/输出      | 384 → 384                                   |
| ONN 数据组织       | 8 个 chunk，每个 196 token                   |
| SLM 数量           | 当前配置为 5 层   **可以改为1层**         |
| 光学读出           | 单探测器光强读出，减去可学习强度偏移         |
| 反馈               | 开启，固定注入第 n 个 SLM                    |
| 训练损失           | 预测目标特征与真实目标特征之间的平均 L1 损失 |

需要特别区分仓库中同时存在的两条路线：`predictor_type: onn_feedback` 对应本文介绍的新增 Predictor 整体替换方案；而 `vit_transformer + qkv_backend=fsonn_tdm` 对应旧的光学 QKV 替换方案。后者仍然保留电子 Transformer Block，只把其中的 QKV 线性映射换成光学模块，不能与当前新增的 `ONNFeedbackPredictor` 混为一谈。

## 二、原始 V-JEPA 外围结构保持不变

### 2.1 视频到 1568 个时空 token

当前模型输入是一段形状为 `[B, 3, 16, 224, 224]` 的 RGB 视频。空间方向使用 `16×16` patch，因此每一帧被划分为 `14×14=196` 个空间 patch；时间方向使用 `tubelet_size=2`，16 帧被划分为 8 个时间 tubelet。最终得到：

```text
时间 token 数：16 / 2 = 8
空间 token 数：224 / 16 = 14，14 × 14 = 196
总 token 数：8 × 196 = 1568
```

V-JEPA 的 context encoder 根据 `masks_ctxt` 只处理上下文位置，输出上下文特征；target encoder 则对完整 token 序列进行编码，再根据 `masks_tgt` 取出预测目标。两条编码器分支都从 V-JEPA 预训练 checkpoint 中加载，在新增 ONN Predictor 的端到端训练中保持冻结。

在当前代码中，context encoder 和 target encoder 仍然是 ViT-L 结构，单个 token 的特征维度为 1024。也就是说，ONN 并没有改变 V-JEPA 的时空 token 化方式和编码器输出语义，只接管了“根据上下文 token 预测目标 token”这一步。

### 2.2 Predictor 的替换边界

因此当前架构的替换关系是：

```text
原始 V-JEPA：
context encoder → 电子 Transformer Predictor → target feature prediction

当前 ONN 路线：
context encoder → ONNFeedbackPredictor → target feature prediction
```

`ONNFeedbackPredictor` 内部不再包含原始的 12 个电子 Transformer Predictor Block，也不再执行标准 self-attention、电子 QKV、残差连接和 FFN 的完整串联。它使用“线性降维 + 固定位置编码 + 串行 ONN + LayerNorm + 输出 MLP”构成新的预测路径。

## 三、ONNFeedbackPredictor 的输入组织

### 3.1 上下文特征先从 1024 维压缩到 384 维

Predictor 接收的上下文特征为：

```text
ctxt: [B, N_ctxt, 1024]
```

首先经过 `predictor_embed`：

```text
[B, N_ctxt, 1024]
        │ Linear(1024 → 384)
        ▼
[B, N_ctxt, 384]
```

这一步是必要的，因为当前 ONN 的输入维度和输出维度均设置为 384。它对应配置中的：

```yaml
predictor:
  context_dim: 1024
  predictor_dim: 384
  output_dim: 1024

onn:
  input_dim: 384
  output_dim: 384
```

### 3.2 恢复为完整的 1568-token 原始位置序列

原始 Predictor 的输入不是一个已经连续排列好的完整序列，而是由上下文位置索引选出的 token。新增 ONN Predictor 为了适应固定的光学阵列尺寸，先创建一个完整的稠密张量：

```text
dense_input = zeros([B, 1568, 384])
```

然后按照 mask 索引执行两次散射：

```text
masks_ctxt 对应的位置：写入 context_384
masks_tgt  对应的位置：写入可学习 mask_token
未被覆盖的位置：保持为 0
```

这里的 `mask_token` 是一个可学习的 `[1,1,384]` 参数，并被复制到所有 target 位置。它不是真实目标特征，而是告诉 Predictor：“这些位置需要被预测”。真实的 target 特征只用于后面的监督损失，不会作为 ONN 的输入内容泄漏到预测路径中。

代码还会检查 `masks_ctxt` 和 `masks_tgt` 是否存在重复、重叠或越界。如果两个 mask 合并后覆盖了同一个 token，Predictor 会直接报错；这保证每个原始 token 位置只有一种状态：上下文、目标或未覆盖。

### 3.3 加入固定三维正弦位置编码

稠密序列构造完成后，代码加上形状为 `[1,1568,384]` 的三维 sin-cos 位置编码：

```text
[B, 1568, 384] + [1, 1568, 384]
        ▼
[B, 1568, 384]
```

位置编码由 8 个时间位置和每个时间位置下的 `14×14` 空间位置共同生成。它被注册为 buffer，而不是 `nn.Parameter`，因此在训练过程中不会更新。这样做保证了 token 的时空位置含义由固定编码提供，不会被 ONN 的训练过程重新改变。

这一步之后，Predictor 已经形成一个固定长度、固定排列的 1568-token 输入：

```text
dense_input: [B, 1568, 384]
```

## 四、1568 token 到 8 个 ONN chunk

### 4.1 固定划分方式

代码将完整序列直接 reshape 为：

```text
[B, 1568, 384]
        reshape
        ▼
[B, 8, 196, 384]
```

其中每个 chunk 的形状为：

```text
x_chunk: [B, 196, 384]
```

因为每个时间 tubelet 正好对应 196 个空间 patch，所以一个 chunk 可以理解为一个时间片上的完整空间 token 网格。当前实现没有把 8 个 chunk 同时送入 8 套光学系统，而是使用同一个 `FeedbackFSONN` 实例依次处理 8 次：

```text
chunk 0 → 同一个 ONN
chunk 1 → 同一个 ONN
chunk 2 → 同一个 ONN
...
chunk 7 → 同一个 ONN
```

因此，这里的 8 次处理属于时间复用。所有 chunk 共享同一组 SLM 相位参数、输入缩放参数、反馈增益和强度偏移参数；不同 chunk 之间的差异来自各自的输入 token 以及串行反馈状态。

### 4.2 恢复 1568 原位再切 chunk

上下文 token 和目标 token 在编码器输出中通常是通过 mask 索引选出来的，直接拿到的张量只包含被选中的位置，不一定构成完整的时间—空间顺序。如果直接将这些压缩后的 token 切成 196 个一组，就可能把不同时间或不同空间位置错误地拼在一起。

当前实现的顺序是：

```text
mask 索引 token
      │
      ├── scatter 回 1568 个原始时空位置
      │
      ├── 加 1568 个固定位置编码
      │
      └── 按原始顺序 reshape 为 8×196
```

因此每个 ONN chunk 对应明确的时空位置，后面才能把 ONN 输出重新拼回 1568 个原始 token，并准确抽取 `masks_tgt` 所对应的预测结果。

## 五、FeedbackFSONN 的光学计算过程

### 5.1 ONN 的尺寸和光学参数

当前 `onn_feedback_intphys.yaml` 中，新增 ONN 的主要配置为：

```yaml
onn:
  input_dim: 384
  output_dim: 384
  num_slm_layers: 5
  chunk_tokens: 196
  feedback_mode: fixed_middle_phase
  feedback_enabled: true
  feedback_memory_enabled: true
  feedback_memory_alpha: 0.8
  feedback_layer_index: 1
  readout_mode: intensity_minus_learnable_offset
  learnable_intensity_offset: true
  use_differential_detector: false
```

因此 ONN 的核心网格尺寸是：

```text
高度：196，对应一个 chunk 中的 196 个 token
宽度：384，对应每个 token 的 384 个内部特征通道
```

在代码中，每一层 `PhaseSLM` 的可学习相位参数形状为 `[196,384]`。这并不意味着 196 个 token 被分别送入 196 个独立网络，而是把 token 维和特征维组织成一个二维复振幅场，再通过自由空间传播和相位调制实现整体光学变换。

当前光学参数还包括：像素间距 8 μm、波长 532 nm、输入到第一层 SLM 的传播距离 50 mm、相邻 SLM 之间的距离 25 mm、最后一层 SLM 到探测面的距离 50 mm，以及 2 倍的角谱传播 padding。

### 5.2 384 维实数特征到复光场

对于当前 chunk：

```text
x_chunk: [B, 196, 384]
```

代码先使用一个可学习的正尺度参数对输入进行缩放，并将结果裁剪到 `[-1,1]`。随后使用 signed-phase 编码：

```text
特征值为正：相位 0，相应幅度为 |x|
特征值为负：相位 π，相应幅度为 |x|
```

于是实数特征被转成复数场：

```text
实数特征 [B,196,384]
        │ signed-phase encoding
        ▼
复光场 [B,196,384]
```

当前 `input_dim=output_dim=384`，所以不存在额外的宽度补零；如果 ONN 网格宽度大于输入维度，代码才会在最后一个维度补零。

### 5.3 自由空间传播与 SLM 相位调制

复光场首先经过一次 band-limited angular spectrum propagation，到达第一层 SLM。之后依次执行：

```text
输入复光场
    │
    ├── 自由空间传播 50 mm
    ├── SLM1：乘以可学习相位调制
    ├── 自由空间传播 25 mm
    ├── SLM2：乘以可学习相位调制
    ├── 自由空间传播 25 mm
    ├── SLM3：乘以可学习相位调制
    ├── 自由空间传播 25 mm
    ├── SLM4：乘以可学习相位调制
    ├── 自由空间传播 25 mm
    ├── SLM5：乘以可学习相位调制
    └── 自由空间传播 50 mm 到探测面
```

每个 SLM 的基础相位由 `2π·sigmoid(phase_logits)` 得到，确保基础相位位于有限范围内。自由空间传播由 `band_limited_angular_spectrum()` 实现，内部通过二维 FFT、传播传递函数和逆 FFT 计算复场传播，并使用 band limit 抑制采样条件下的非传播频率。

### 5.4 单探测器强度读出

到达探测面后，代码计算复场的模平方：

```text
intensity = |field|²
```

输出保持为：

```text
y_chunk: [B,196,384]
```

如果启用 `learnable_intensity_offset`，则从强度输出中减去一个形状为 `[1,1,384]` 的可学习偏移：

```text
y_chunk = intensity - intensity_offset
```

当前新增反馈 ONN 使用单探测器，不使用旧版的正负双探测器、正负增益或差分探测路径。代码会拒绝这些旧配置字段，避免把旧版 QKV 光学读出逻辑混入新的 Predictor。

## 六、跨 chunk 的反馈和记忆机制

8 个 chunk 共享同一个 ONN，但它们不是相互独立的前向计算。第 $t$ 个 chunk 在探测器处产生的输出，经过光强读出、可学习强度偏移、LayerNorm 和记忆更新后，形成作用于下一个 chunk 的相位反馈。

这里的反馈不是把前一个 chunk 的输出直接加到下一个 chunk 的输入上，而是按照下面的完整链路生成 SLM 相位增量：

$$
I_t
=
\left|E_t^{\mathrm{det}}\right|^2
$$

其中 $E_t^{\mathrm{det}}$ 表示第 $t$ 个 chunk 到达探测器的复光场，$I_t$ 是单探测器读出的光强。

代码不使用正负双探测器差分，而是减去一个 384 维的可学习光强偏移量 $B$：

$$
Y_t=I_t-B
$$

其中：

$$
B\in\mathbb{R}^{1\times1\times384}
$$

$B$ 会沿 batch 维和 token 维广播，因此每个输出通道都有一个独立的可学习偏移。

随后，对减去偏移后的输出执行不带可学习仿射参数的 LayerNorm：

$$
R_t
=\operatorname{LN}(Y_t)
=\operatorname{LN}(I_t-B)
$$

对于 batch 中的第 $b$ 个样本、空间 token $n$ 和特征通道 $c$，其归一化形式为：

$$
\mu_{t,b,n}
=\frac{1}{384}\sum_{c=1}^{384}Y_{t,b,n,c}
$$

$$
\sigma^2_{t,b,n}
=\frac{1}{384}\sum_{c=1}^{384}
\left(Y_{t,b,n,c}-\mu_{t,b,n}\right)^2
$$

$$
R_{t,b,n,c}
=\frac{Y_{t,b,n,c}-\mu_{t,b,n}}
{\sqrt{\sigma^2_{t,b,n}+\varepsilon}}
$$

因此 $R_t$ 保持与 ONN 输出相同的形状：

$$
R_t\in\mathbb{R}^{B\times196\times384}
$$

### 6.1 第一个 chunk 的记忆初始化

第一个 chunk 没有历史状态，因此归一化后的第一个输出直接作为初始记忆：

$$
M_1=R_1
$$

第一个 chunk 不使用前序反馈，因此：

$$
\Delta\Phi_1=0
$$

### 6.2 后续 chunk 的记忆状态更新

从第二个 chunk 开始，当前输出以指数平滑的方式更新历史记忆：

$$
M_t
=\alpha M_{t-1}+(1-\alpha)R_t,
\qquad t\ge2
$$

当前配置为：

$$
\alpha=0.8
$$

所以实际更新公式为：

$$
\boxed{
M_t=0.8M_{t-1}+0.2R_t,
\qquad t\ge2
}
$$

代入当前 ONN 输出的光强、偏移和归一化过程后，记忆更新可以完整写为：

$$
\boxed{
M_t
=0.8M_{t-1}
+0.2\operatorname{LN}\left(\left|E_t^{\mathrm{det}}\right|^2-B\right),
\qquad t\ge2
}
$$

记忆状态不是一个单独的标量，而是完整保留 batch、196 个 token 和 384 个通道的张量：

$$
M_t\in\mathbb{R}^{B\times196\times384}
$$

### 6.3 记忆状态转换为反馈相位

在第 $t+1$ 个 chunk 到来之前，系统读取前一个 chunk 形成的记忆 $M_t$，先进行反馈归一化，再通过可学习反馈增益 $K$ 生成相位增量：

$$
Q_t=\operatorname{LN}(M_t)
$$

$$
\boxed{
\Delta\Phi_{t+1}
=\frac{\pi}{2}
\tanh\left(K\operatorname{LN}(M_t)\right)
}
$$

其中：

* $K$ 是可学习的反馈增益；
* $\pi/2$ 是当前配置中的最大反馈相位幅度；
* $\tanh$ 将反馈限制在有限范围内。

因此：

$$
-\frac{\pi}{2}
\le \Delta\Phi_{t+1}
\le \frac{\pi}{2}
$$

并且：

$$
\Delta\Phi_{t+1}\in\mathbb{R}^{B\times196\times384}
$$

### 6.4 相位反馈注入第 2 个 SLM

当前配置为：

```yaml
feedback_layer_index: 1
```

由于代码索引从 0 开始，`feedback_layer_index: 1` 对应物理上的第 2 个 SLM。第 $t+1$ 个 chunk 在第 2 个 SLM 上使用的最终相位为：

$$
\boxed{
\widetilde{\Phi}_{2,t+1}
=\Phi_2+\Delta\Phi_{t+1}
}
$$

代入记忆到相位的转换公式，可以得到：

$$
\boxed{
\widetilde{\Phi}_{2,t+1}
=\Phi_2
+\frac{\pi}{2}
\tanh\left(K\operatorname{LN}(M_t)\right)
}
$$

其中 $M_t$ 又由当前及之前 chunk 的 ONN 输出递推得到：

$$
M_1=\operatorname{LN}\left(\left|E_1^{\mathrm{det}}\right|^2-B\right)
$$

$$
M_t
=0.8M_{t-1}
+0.2\operatorname{LN}\left(\left|E_t^{\mathrm{det}}\right|^2-B\right),
\qquad t\ge2
$$

所以，从第一个 ONN 输出开始，到下一个 chunk 的 SLM2 相位反馈，完整公式链为：

$$
\boxed{
\left|E_t^{\mathrm{det}}\right|^2
\rightarrow
\left(\left|E_t^{\mathrm{det}}\right|^2-B\right)
\rightarrow
R_t=\operatorname{LN}\left(\left|E_t^{\mathrm{det}}\right|^2-B\right)
\rightarrow
M_t
\rightarrow
\Delta\Phi_{t+1}=\frac{\pi}{2}\tanh\left(K\operatorname{LN}(M_t)\right)
\rightarrow
\widetilde{\Phi}_{2,t+1}=\Phi_2+\Delta\Phi_{t+1}
}
$$

因此，当前反馈过程可以概括为：

$$
\boxed{
\text{第 }t\text{ 个 chunk 的探测器光强}
\rightarrow
\text{减去可学习偏移 }B
\rightarrow
\text{LayerNorm}
\rightarrow
\text{记忆更新}
\rightarrow
\text{反馈相位增量}
\rightarrow
\text{第 }t+1\text{ 个 chunk 的第 2 个 SLM}
}
$$

当前不是把反馈同时施加到所有 SLM，而是仅修改第 2 个 SLM 的相位：

$$
\widetilde{\Phi}_{2,t+1}=\Phi_2+\Delta\Phi_{t+1}
$$

其余 SLM 保持基础相位：

$$
\widetilde{\Phi}_{1,t+1}=\Phi_1,
\qquad
\widetilde{\Phi}_{3,t+1}=\Phi_3,
\qquad
\widetilde{\Phi}_{4,t+1}=\Phi_4,
\qquad
\widetilde{\Phi}_{5,t+1}=\Phi_5
$$

## 七、ONN 输出如何恢复成目标预测

8 个 chunk 分别完成 ONN 前向后，输出被沿 token 维拼接：

```text
8 × [B,196,384]
        │ cat(dim=1)
        ▼
dense_output: [B,1568,384]
```

随后代码执行一次 `predictor_norm`，再使用 `masks_tgt` 从完整输出中抽取目标位置：

```text
[B,1568,384]
        │ gather(masks_tgt)
        ▼
pred_tgt_384: [B,N_tgt,384]
```

最后通过输出 MLP：

```text
[B,N_tgt,384]
        │ Linear(384 → 384)
        │ GELU
        │ Linear(384 → 1024)
        ▼
pred_tgt_1024: [B,N_tgt,1024]
```

这个 `pred_tgt_1024` 才是新增 ONN Predictor 的最终输出。它与冻结 target encoder 在相同 `masks_tgt` 位置得到的真实目标特征进行比较。换句话说，ONN 内部只负责生成 384 维预测表示，最后的 MLP 负责把它恢复到原始 V-JEPA 目标特征空间。

## 八、训练数据流

### 8.1 每个训练 batch 的输入

当前 `onn_feedback_intphys.yaml` 使用 `end_to_end_jepa` 模式和 `onn_feedback` Predictor。每个训练样本是一段 16 帧视频，代码从训练数据中生成或读取上下文 mask 和目标 mask。当前配置还设置了：

```yaml
mask_mode: unified_random
data_split:
  num_train_videos: 800
  num_val_videos: 100
  split_seed: 42
```

`unified_random` 的含义是，同一个 batch 内使用统一的随机 mask 位置，使所有样本具有相同的固定 token 布局，更适合当前固定的 1568-token 到 8×196-token 组织方式。

### 8.2 冻结分支产生监督目标

训练时，代码先执行两个冻结的编码器分支：

```text
视频 clip [B,3,16,224,224]
        │
        ├── target encoder，完整 token
        │       └── target feature [B,1568,1024]
        │               └── LayerNorm
        │                       └── 取 masks_tgt
        │
        └── context encoder，只取 masks_ctxt
                └── context [B,N_ctxt,1024]
```

目标分支使用 `torch.no_grad()`，并在特征维度上进行 LayerNorm。目标特征经过 `masks_tgt` 选取后，作为损失计算中的监督信号；它不会作为 `ONNFeedbackPredictor` 的真实输入。

### 8.3 新 Predictor 生成预测目标

上下文特征送入新增 Predictor 后，按前述流程完成：

```text
context [B,N_ctxt,1024]
        │
        ├── 1024→384
        ├── scatter 到 1568 原位
        ├── target 位置填充 mask token
        ├── 加固定位置编码
        ├── 8 次串行 FeedbackFSONN
        ├── 抽取目标位置
        └── 384→1024
```

需要注意，`ONNFeedbackPredictor.forward()` 中的 `tgt` 参数被显式忽略。它保留这个参数只是为了兼容原始 V-JEPA Predictor 的调用接口；当前新增 Predictor 真正使用的是 `ctxt`、`masks_ctxt` 和 `masks_tgt`。真实目标特征只在损失计算阶段出现。

### 8.4 训练损失和可更新参数

当前配置 `loss_exp=1.0`，因此 `_compute_jepa_loss()` 实际计算的是预测特征和目标特征之间的平均 L1 损失：

```text
L = mean(|pred_tgt_1024 - target_tgt_1024|)
```

训练时冻结：

```text
context encoder
target encoder
```

新增 Predictor 内部可更新：

```text
predictor_embed
mask_token
FeedbackFSONN 的 SLM 相位参数
输入缩放参数
反馈增益参数
强度偏移参数
predictor_norm
output_mlp
```

固定不更新：

```text
predictor_pos_embed
```

因此当前实验训练的并不是一个已经加载了原始 Predictor 权重的光学替代品，而是一个使用冻结 V-JEPA 编码器作为输入和目标、从头学习新增预测路径的 ONN Predictor。

### 8.5 验证和 checkpoint 选择

训练脚本每个 epoch 都会分别运行 train 和 val。每个阶段都使用冻结的 context encoder、target encoder 和当前 Predictor，计算同样的 JEPA 特征预测损失。代码根据 `val_jepa_loss` 是否下降来保存 best checkpoint，而不是根据 IntPhys 的 relative accuracy 或 absolute accuracy 选择 checkpoint。

训练结束后，脚本可以调用最终评测流程，对 Dev 和 Test 进行前向计算。checkpoint 中同时记录 Predictor 状态、ONN 配置、mask 模式、token 数、chunk 数、反馈层、反馈增益和预训练 checkpoint 路径，便于后续确认实验使用的具体架构。

## 九、评测数据流

### 9.1 视频窗口和未来预测

评测阶段不使用 test 标签参与前向计算。对于每个视频，代码根据 `frame_step` 从原始帧序列中构造多个长度为 16 帧的滑动窗口；每个窗口都使用不同的上下文长度，例如当前配置中的 `[2,4,6,8,10]`。

对每个窗口，评测流程仍然是：

```text
视频窗口
   │
   ├── context encoder → context feature
   ├── target encoder → target feature
   ├── ONNFeedbackPredictor(context, masks)
   └── 对预测特征和目标特征计算 L1 surprise
```

每个目标窗口最终得到一个 surprise。当前代码在 token 维和特征维上对逐元素 L1 差异求平均，形成该窗口的预测误差。

### 9.2 从窗口 surprise 到视频分数

对于单视频指标，代码会使用一个视频中所有窗口 surprise 的最大值或平均值。当前新增 Predictor 的输出本身不是 possible/impossible 分类概率，而是目标特征预测误差；评测时通过方向转换得到 plausibility：

```text
plausibility = -surprise
```

因此：

```text
surprise 越小  → 视频越符合预测 → plausibility 越高
surprise 越大  → 视频越违背预测 → plausibility 越低
```

### 9.3 相对和绝对指标

IntPhys 的官方四元组包含 2 个 possible 视频和 2 个 impossible 视频。当前代码会保留完整四元组结构，并计算两类指标：

```text
相对指标：比较同一四元组中 possible 两个视频的 plausibility 总和与 impossible 两个视频的 plausibility 总和。

绝对指标：把所有视频放在一起，用 plausibility 计算 AUROC；官方代码把绝对错误率记录为 1-AUROC。
```

需要注意，仓库中还保留了一个通用 `compute_metrics()` 路径，它会用 possible 视频的 90% 分位误差构造固定阈值，并记录 `Absolute Accuracy (max)` 和 `Best Absolute Accuracy (max)`。而 `compute_official_metrics()` 使用的是官方四元组结构和 `1-AUC` 定义。写实验报告时应明确标注究竟使用的是阈值准确率、AUROC，还是官方的 `1-AUROC` 错误率，不能把这些数值混写。

## 十、当前新架构与旧版光学 QKV 路线的区别

仓库中的 [`predictor.py`](https://github.com/wangcoco118/jepa_onn/blob/main/JEPA_ONN/src/models/predictor.py) 同时保留了原始电子 Predictor、旧版 `TimeDivisionFSONN` 和新增 `ONNFeedbackPredictor`，因此阅读代码时容易混淆。

| 对比项                         | 旧版光学 QKV 路线                     | 当前新增 ONN Predictor 路线                      |
| ------------------------------ | ------------------------------------- | ------------------------------------------------ |
| 替换范围                       | Transformer Predictor 内部的 QKV 映射 | 整个 Predictor                                   |
| 主要类                         | `TimeDivisionFSONN`                 | `FeedbackFSONN` + `ONNFeedbackPredictor`     |
| 是否保留电子 Transformer Block | 保留                                  | 不使用原始 Predictor Block                       |
| ONN 输入                       | 通常是 523-token 时隙、384 维         | 固定为 196-token chunk、384 维                   |
| ONN 输出                       | 可配置为 QKV 的 1152 维               | 384 维特征输出                                   |
| 读出方式                       | 旧版双路正负探测和差分逻辑            | 单探测器强度减可学习偏移                         |
| chunk 组织                     | 旧版时隙划分                          | 1568 = 8×196，串行复用同一 ONN                  |
| 跨 chunk 反馈                  | 旧版接口较简单                        | 显式反馈状态和指数平滑记忆                       |
| 最终输出                       | 继续进入电子 Attention                | ONN 输出经 LayerNorm、抽取 target、MLP 回到 1024 |

因此，当前研究中如果讨论“新增的 ONN Predictor”，准确的表述应是：

> 在冻结的 V-JEPA ViT-L context/target encoder 之间，使用一个固定 1568-token 布局、8×196 串行时间复用、带 SLM 相位反馈和记忆状态的自由空间 ONN Predictor，替换原始电子 Transformer Predictor，并通过输出 MLP 将 ONN 的 384 维表示映射回 1024 维目标特征空间。

|  |  |
| - | - |
