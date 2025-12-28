**“Revision Notes: Core CNN Operations & Hyperparameters”**

---

# 🔁 Revision Notes: Core CNN Operations & Hyperparameters

This section provides **quick definitions and intuition** for the main building blocks used in convolutional neural networks.

---

## 1️⃣ Convolution

### Definition

**Convolution** is a linear operation where a small matrix (kernel/filter) slides over an input and computes a **weighted sum (dot product)** at each spatial location.

### Mathematical view

$$

[
\text{output}(i,j) = \sum_{u,v} \text{input}(i+u, j+v) \cdot \text{kernel}(u,v)
]

$$

### Intuition

* Detects **local patterns** (edges, textures, shapes)
* Weight sharing makes it **parameter efficient**
* Same kernel is applied across all spatial locations

### Key takeaway

> Convolution extracts **features**, not pixels.

---

## 2️⃣ Padding (We have seen it already in last class and today's class)

### Definition

**Padding** adds extra pixels (usually zeros) around the input image before convolution.

### Why padding is used

1. Prevents spatial shrinkage
2. Preserves border information
3. Controls output size

### Common types

* **Valid padding**: no padding → output shrinks
* **Same padding**: output spatial size ≈ input size
* **Zero padding**: most common

### Intuition

> Padding gives the kernel “room to move” near the edges.

---

## 3️⃣ Stride (It's a jump of convolution - we have seen in last class during the animation)

### Definition

**Stride** is the number of pixels the kernel moves at each step during convolution.

### Effects of stride

* Larger stride → smaller output
* Reduces spatial resolution
* Acts as **implicit downsampling**

### Example

* Stride = 1 → dense scanning
* Stride = 2 → skip every alternate pixel

### Intuition

> Stride trades **resolution for efficiency**.

---

## 4️⃣ Dilation (We have seen it, we will see in future in the place where we see receptive field)

### Definition

**Dilation** spaces out kernel elements by inserting gaps between them.

### Mathematical meaning

* Dilation enlarges the **receptive field** without increasing parameters

### Why dilation is useful

* Captures wider context
* Useful in segmentation, time-series, audio

### Intuition

> Dilation lets the kernel “see farther” without growing in size.

---

## 5️⃣ Pooling (We will see this in next class)

### Definition

**Pooling** is a non-learnable downsampling operation applied to feature maps.

### Common pooling types

* **Max Pooling** → keeps strongest activation
* **Average Pooling** → keeps average response

### Purpose

* Reduce spatial dimensions
* Introduce translation invariance
* Reduce computation

### Important distinction

> Pooling has **no learnable parameters**, unlike convolution.

---

## 6️⃣ Relationship Between These Components

| Component   | Role                      |
| ----------- | ------------------------- |
| Convolution | Feature extraction        |
| Padding     | Boundary handling         |
| Stride      | Resolution control        |
| Dilation    | Context expansion         |
| Pooling     | Downsampling & invariance |

---

## 🧠 Unified Mental Model (We will see this in future)

A typical CNN block performs:

```
Convolution → Activation → (Optional) Pooling
```

With:

* Padding controlling **where**
* Stride controlling **how often**
* Dilation controlling **how far**
* Pooling controlling **how much is retained**

---

## 📌 Final Revision Insight

> **Convolution learns what to look for.**
> **Padding, stride, dilation, and pooling decide how and where to look.**

This separation is why CNNs are both:

* Expressive
* Computationally efficient

---
