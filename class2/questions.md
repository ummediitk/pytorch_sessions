Below is a **clean, classroom-ready rephrasing** with clearer language, consistent terminology, and unambiguous expectations — while keeping the *spirit* of your original questions intact.

I’ve also slightly tightened the phrasing so students focus on **concepts (shape, reduction, slicing, visualization)** rather than guessing intent.

---

# Assignment: Tensor Shapes, Reductions, and Visualization in PyTorch

### **Important Instructions**

* Use **PyTorch documentation**, **Google**, and **StackOverflow** to solve these problems.
* **Do NOT use LLMs or ChatGPT** to solve the questions (this is for your own learning).
* If you get stuck:

  * Ask me
  * Ping classmates
  * Discuss approaches
    **But do not use LLMs.**

---

## **Question 1 — Reproducible Random Tensor**

Generate **120 random numbers** using PyTorch with a fixed random seed:

* Use `torch.manual_seed(123)`
* Store the result as a single 1-D tensor

---

## **Question 2 — Reshaping with PyTorch and Einops**

Reshape the tensor from **Question 1** into a 4-D tensor with the following dimensions:

* **Batch = 5**
* **Channels = 2**
* **Height = 4**
* **Width = 3**

Do this **in two ways**:

1. Using PyTorch’s `.reshape()`
2. Using **einops**

Verify that both methods produce the same shape.

---

## **Question 3 — Statistics per Batch and Channel**

Using the tensor from **Question 2**, compute the following statistics:

* Mean
* Standard deviation
* Variance

Compute these **for each batch and each channel**.

> Use `einops` and PyTorch reduction operations (`torch.mean`, etc.)
> Think carefully about **which dimensions to reduce**.

---

## **Question 4 — Statistics per Batch, Channel, and Row**

Extend **Question 3** by computing:

* Mean
* Standard deviation
* Variance

This time compute them **for each batch, each channel, and each row (height)**.

> This requires a deeper understanding of tensor dimensions and reductions.

---

## **Question 5 — 2D Visualization**

Take the tensor from **Question 1** and:

1. Reshape it into a **2-D tensor of shape (15, 8)**

   * 15 rows
   * 8 columns
2. Convert it to a NumPy array
3. Plot it using **matplotlib**

Observe and comment:

* What does the visualization look like?
* Does the randomness appear uniform?

---

## **Question 6 — Binary Pattern Tensor**

Using the **same seed as Question 1**, create a **10 × 10 tensor** such that:

* Values alternate between **1 and 0**
* The element at position `(0, 0)` is **1**
* Adjacent cells must have opposite values

---

## **Question 7 — Visualizing the Pattern**

Convert the tensor from **Question 6** to a NumPy array and plot it using **matplotlib**.

* What pattern do you observe?
* Does it resemble a chessboard?

---

## **Question 8 — Colored Visualization**

Modify the visualization from **Question 7** so that:

* The pattern appears in **color** (red, green, or blue)
* Instead of black-and-white

> Hint: Think about how matplotlib maps values to colors.

---

## **Question 9 — Column Slicing and Masking**

Using the 2-D tensor from **Question 5**:

1. Slice **every alternate column**
2. Replace those columns with zeros
3. Convert the result to a NumPy array
4. Plot it again using **matplotlib**

Compare this plot with the original from Question 5.

---

## **Question 10 — Concept Reflection (Mandatory)**

Write a short notebook explaining:

* What you learned in this class
* Your understanding of:

  * Tensor shapes
  * Reshaping
  * Reductions
  * Visualization

Once done:

* Paste your explanation into ChatGPT **only to verify your understanding**, not to generate answers.

---

## **Bonus Question — Symbol Generation (Fun Task)**

Create **9 × 9 binary grids** representing mathematical symbols.

Example: **Plus (+)**

* Entire **5th row** = 1
* Entire **5th column** = 1
* All other cells = 0

Try implementing:

* Plus (+)
* Minus (−)
* Multiplication (×)

Visualize them using matplotlib.

---
