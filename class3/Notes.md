

# PyTorch – Class 3

## Numerical Stability, Softmax, F1/Dice, and `nn.Module` Internals

---

## 1. What We Covered Today

* Why **numerical instability** matters in ML programs
* Two classic instability examples:

  * **Softmax**
  * **F1 Score / Dice Score**
* Understanding **feature vectors**
* Proper implementation of **Softmax**
* `nn.Module` basics:

  * `__init__` vs `forward`
  * Parameters vs Buffers
* Why PyTorch APIs are written the way they are

---

## 2. Numerical Instability – Why Should We Care?

Numerical instability occurs when:

* Floating-point overflow / underflow happens
* Division by zero or near-zero values occurs
* Small rounding errors explode during training

### Why this matters:

* Model outputs become `inf` or `nan`
* Loss suddenly becomes undefined
* Gradients explode or vanish
* Kaggle / graders fail even when math is “correct”

---

## 3. Feature Vector Clarification

### Important conceptual point:

> **Feature Vector ≠ Column Vector**

In PyTorch:

* A feature vector is **1-D** or **row-wise**
* Shape matters more than “column vs row” intuition from math books

Example:

```python
t1.shape  # (batch_size, num_features)
```

---

## 4. Softmax – The First Instability Example

### Naive Softmax Formula
$$
[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
]
$$
Here’s a **clean, precise phrasing** you can directly paste into your earlier markdown. I’ve kept the tone consistent with class-notes style and added the **numerically stable softmax** explicitly.

---

## Why Is It Called **Softmax**?

The operation is called **softmax** because it behaves like a *soft* (differentiable) version of the **max** operator.

When we apply softmax to a vector ( x ), the output is a probability vector where:

* The largest element in ( x ) gets the **highest weight**
* Other elements are **suppressed but not discarded**
* The result is smooth and differentiable (unlike hard max)

A useful intuition:

> **If we take the softmax of a vector and compute the dot product of the softmax vector with the original vector, the result is very close to the maximum value of the vector.**

Mathematically, if
$$
[
p = \text{softmax}(x)
]
$$

then

$$
[
p^\top x \approx \max(x)
]
$$

This shows that **softmax concentrates probability mass near the maximum**, but still keeps gradient information from all elements — hence the name *soft*-max.

---

## Numerically Stable Softmax (Safe Formula)

The naive softmax implementation:

$$
[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
]
$$
is **numerically unstable** for large values of ( x ), because ( e^{x_i} ) can overflow.

### Stable (Safe) Softmax Formula

To make softmax numerically stable, we subtract the maximum value before exponentiation:
$$
[
\text{softmax}(x_i) =
\frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
]
$$
This transformation:

* Does **not** change the output probabilities
* Prevents overflow and underflow
* Is what PyTorch and all major ML libraries use internally

### PyTorch Reference

```python
nn.Softmax(dim=...)
```

uses this **numerically stable implementation** by default.

---


### Problem

If `x` contains large values:

* `exp(x)` → overflow
* Result becomes `inf` or `nan`

---

### Stable Softmax Trick (Log-Sum-Exp Trick)

Key idea:

> **Subtract the maximum value before exponentiation**

```python
def stable_softmax(x):
    exp_x = torch.exp(x - x.max())
    return exp_x / exp_x.sum()
```

### Why this works:

* Shifting values does **not** change relative probabilities
* Prevents overflow
* PyTorch internally does this for you

---

### PyTorch Implementation

```python
nn.Softmax(dim=1)
```

Internally uses **numerically stable** computation.

---

## 5. Hardmax vs Softmax

* **Softmax** → probability distribution (smooth, differentiable)
* **Hardmax** → one-hot output (non-differentiable)

Hardmax is **not used in training**, only sometimes in inference.

---

## 6. F1 Score / Dice Score – Second Instability Example

### Naive F1 Formula

[
F1 = \frac{2PR}{P + R}
]

### Problem:

* Precision or Recall may be undefined
* Division by zero possible
* Different invalid cases depending on TP, FP, FN

---

### Stable Dice / F1 Formula

```python
dice = (2 * tp) / (2 * tp + fp + fn + eps)
```

### Why this is better:

* Uses raw counts instead of ratios
* Only **one** invalid edge case
* Guaranteed range: **[0, 1]**
* Used in:

  * Image segmentation
  * Medical ML
  * Kaggle competitions

---

## 7. Key Lesson on Numerical Reformulation

> **Same mathematical meaning ≠ same numerical behavior**

Rewriting formulas:

* Improves stability
* Avoids undefined intermediate values
* Makes code robust for real-world data

---

## 8. `nn.Module` – Mental Model

Every PyTorch model follows this structure:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        ...
```

## `Dropout` (The confusion)

---

## Dropout

### What Dropout does

**Dropout** is a regularization technique used during training to reduce overfitting by **randomly zeroing out activations**.

If a layer output is (x), dropout produces:
$$
[
\tilde{x} = m \odot x,\quad m_i \sim \text{Bernoulli}(1-p)
]
$$
where:

* (p) = dropout probability (fraction of units dropped)
* (m) is a random mask (0/1)
* $(\odot)$ is elementwise multiplication

During training, PyTorch uses **inverted dropout**, so surviving activations are scaled by (1/(1-p)) to keep expected magnitude consistent.

---

### Important correction (mistake from my side)

I initially described dropout as if it “drops elements in rows” or some structured chunk — that’s **not correct** for standard dropout.

✅ **Standard dropout drops individual elements independently** (each element has its own probability of being zeroed).

Because we usually apply dropout over many elements, it can **visually appear** like drops are “evenly distributed across rows,” but that is just a statistical illusion due to large numbers. In reality:

* there is **no guarantee** that each row gets the same number of dropped elements
* the mask is **random at the element level**

**Devki** pointed this out and helped correct the understanding — thanks to Devki for catching it.

---

### Dropout in CNNs: 1D vs 2D vs channels (RGB)

Dropout behavior depends on *what dimension you want to regularize*:

#### 1) 1D case (single channel / no RGB)

For vectors or 1D feature tensors, applying standard dropout is usually fine because:

* each element corresponds to a feature activation
* dropping random elements makes sense and is commonly used in MLP heads / embeddings / classifier layers

#### 2) 2D case with images (RGB / multi-channel feature maps)

For CNN feature maps, naive elementwise dropout can sometimes be suboptimal because:

* adjacent pixels/features are highly correlated
* dropping random pixels may not regularize in a meaningful way

So you often prefer **structured dropout** in CNNs:

* **`nn.Dropout2d` / Spatial Dropout**: drops **entire channels** (feature maps) rather than individual pixels
  This forces the network not to rely too much on any single feature map.
* With RGB inputs or intermediate feature maps, it’s important to be clear whether you are dropping:

  * random **elements**, or
  * entire **channels**, or
  * blocks/regions (less common, but exists)

**Takeaway:**
In CNNs, be careful: sometimes the right regularizer is “drop a channel / feature map,” not “drop random pixels.”

---

## Convolution conventions: NumPy vs PyTorch

A subtle but important implementation detail:

### Two common conventions

* Many signal-processing texts define convolution as:
$$
  [
  y[n] = \sum_k x[k],h[n-k]
  ]
$$
* But many deep learning libraries implement what is technically **cross-correlation**:
$$
  [
  y[n] = \sum_k x[k],h[n+k]
  ]
$$
### Practical difference

* **NumPy convolution** is typically aligned with the classical definition (kernel indexed with (n-k), effectively “flipping” the kernel).
* **PyTorch Conv layers** (and most DL frameworks) use the **cross-correlation-style** form (kernel not flipped in the same way).

So you’ll often see this described informally as:

* NumPy: $(x[n] * h[n-k])$
* PyTorch: $(x[n] * h[n+k])$

### Why it doesn’t matter in deep learning training

Even though these are different operations mathematically, in neural networks the kernel weights are **learned**, so the model will learn the “flipped” version if needed. Therefore:

* it does **not** reduce model capacity
* it does **not** hurt training
* it’s mainly a **convention difference**

---

## Kernels as learned parameters vs hand-crafted feature engineering

### Modern deep learning

In CNNs today:

* kernels/filters are **trainable parameters**
* learned end-to-end via backpropagation

### Before AlexNet (and early vision era)

Historically, many practitioners did explicit **feature engineering**:

* designing kernels by hand for edge detection, corners, texture, etc.
* examples include classical filters like Sobel / Prewitt / Laplacian-of-Gaussian / Gabor filters
* pipelines were often: “hand-designed features → classifier”

AlexNet (and deep CNNs afterward) popularized the idea that:

> instead of manually inventing kernels, we can learn them automatically from data at scale.

---

## 9. `__init__` vs `forward`

### `__init__`

Used for:

* Parameters (`nn.Linear`, `nn.Conv2d`, etc.)
* Trainable tensors
* Fixed configuration

### `forward`

Used for:

* Actual computation
* Data-dependent operations
* Logic that runs every batch

---

### Rule of Thumb

| Goes in `__init__` | Goes in `forward` |
| ------------------ | ----------------- |
| Weights            | Input data        |
| Biases             | Temporary tensors |
| Hyperparameters    | Batch logic       |
| Buffers            | Conditional flow  |

---

## 10. Parameters vs Buffers

### Parameters

```python
nn.Parameter(...)
```

* Trainable
* Appear in `model.parameters()`
* Saved in checkpoints

---

### Buffers

```python
self.register_buffer("name", tensor)
```

* Not trainable
* Saved in model state
* Used for constants (e.g., running mean, masks)

---

## 11. Why PyTorch Enforces This Design

* Enables:

  * Automatic differentiation
  * Device movement (`.to(device)`)
  * Model saving/loading
  * Distributed training
* Prevents silent bugs
* Makes models reproducible

---

## 12. Big Picture Takeaways

1. **Numerical stability is not optional**
2. Stable math > pretty math
3. PyTorch APIs encode years of battle-tested tricks
4. Understanding *why* formulas are written a certain way matters
5. `nn.Module` structure is about **discipline**, not convenience

## 13. Implementation of FFT in pytorch for convolution

Good catch — that **definitely belongs** in the notes.
Below is a **clean, paste-ready Markdown addition** that fits naturally after the convolution section. I’ve kept it conceptually correct, careful with wording, and honest about what’s usually hidden from tutorials.

---

## Convolution in Practice: FFT vs “Sliding Window”

### How convolution is usually taught

In most tutorials, convolution is explained as:

* sliding a kernel over the input
* multiplying and summing local neighborhoods

This view is **conceptually correct**, but it is **not how high-performance libraries implement convolution internally**.

---

### How convolution is implemented in PyTorch (in practice)

In real-world deep learning frameworks (including PyTorch), convolution is **often implemented using FFT-based methods**, not explicit nested loops.

This is because of a fundamental result from signal processing:

> **Convolution in the time (or spatial) domain is equivalent to multiplication in the frequency domain.**

Mathematically:
$$
[
\mathcal{F}(x * h) = \mathcal{F}(x)\cdot \mathcal{F}(h)
]
$$
where:

* $( * )$ denotes convolution
* $( \mathcal{F} )$ is the Fourier Transform

So instead of:

1. Sliding the kernel across the input
2. Doing many multiplications per position

the computation can be:

1. Transform input and kernel to frequency domain (FFT)
2. Multiply them elementwise
3. Transform back using inverse FFT

---

### Why FFT-based convolution is used

* Direct convolution is (O(N^2)) (or worse in higher dimensions)
* FFT-based convolution reduces this to approximately:
$$
  [
  O(N \log N)
  ]
$$
* This is **much faster** for:

  * large kernels
  * large feature maps
  * deeper layers

---

### Why this is usually not discussed in tutorials

* FFT-based convolution is:

  * mathematically heavy
  * implementation-dependent
  * backend-specific (cuDNN, MKL, etc.)
* It distracts from the **learning objective** (what convolution *does*)
* Frameworks automatically choose the fastest method internally

So most teaching material focuses on:

* the sliding-window intuition
* shapes, padding, stride, and channels

…while the FFT details remain hidden.

---

### Important clarification

* PyTorch does **not always** use FFT
* The backend dynamically chooses between:

  * direct convolution
  * im2col + GEMM
  * FFT-based convolution

The choice depends on:

* tensor size
* kernel size
* hardware (CPU vs GPU)
* memory constraints

From the user’s perspective:

> **You write `nn.Conv2d`, and PyTorch decides the optimal algorithm.**

---

### Big-picture takeaway

* Sliding-window convolution is a **conceptual model**
* FFT-based convolution is a **performance optimization**
* Both compute the same mathematical operation
* The learning process is unaffected by the implementation choice


---

## 14. What This Enables You To Do

After this class, you should be able to:

* Debug `nan` / `inf` issues
* Recognize unstable formulas
* Write production-safe metrics
* Understand PyTorch internals better
* Read research / library code with clarity

