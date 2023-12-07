
## Deep Learning Question

Topic: LSTM Networks

1) Considering the LSTM architecture with the following description,

   1. **Components**:
       - **U**: A linear transformation mapping input `x_t` to a `d`-dimensional vector, defined as `U = W_u x_t + b_u`. Used as input for the `σ` and `s` gates.
       - **μ**: Another linear transformation mapping `x_t` to a `d`-dimensional vector, defined as `μ = W_μ x_t + b_μ`. Used as input for the `β` gate.
       - **W**: A linear transformation mapping the previous hidden state `h_t-1` to a `3d`-dimensional vector, defined as `W = W_w h_t-1 + b_w`. Split into `W_σ`, `W_s`, and `W_β` for the respective gates.
       - **x_t**: The input vector at time step `t`.

   2. **Gates**:
       - **σ (Forget Gate)**: Decides how much of the previous cell state `c_t-1` to retain or discard, defined as `σ = sigmoid(U + W_σ)`. Used to update the cell state.
       - **s (Input Gate)**: Decides how much of the new information from `x_t` to add to the cell state, defined as `s = sigmoid(U + W_s)`. Used to update the cell state.
       - **β (Output Gate)**: Decides how much of the cell state to output as the hidden state, defined as `β = sigmoid(μ + W_β)`. Used to update the hidden state.

   3. **Output**: The hidden state `h_t` is the output, used for tasks such as classification, regression, or generation. The cell state `c_t` is passed to the next LSTM cell, along with `x_t+1` and `h_t`, forming a recurrent neural network for sequential data processing.

Answer the following questions based on the explained architecture.

(a) Write the equations for all the output variables of this cell.

Calculate: $\frac{\partial{h_T}}{x_i}$

Also calculate: $\frac{\partial{h_T}}{h_i}$

(b) Does this architecture face the problem of Gradient Vanishing or Gradient Explosion? Explain and provide a solution for it.

(c) Which optimization algorithm does this cell resemble? Explain how this algorithm works.

## Deep Learning Question

Topic: Recurrent Neural Networks

Consider the architecture of a recurrent network (RNN) described as follows:

```
Network Composition:
- Cells: The network consists of multiple cells that process calculations at discrete time steps denoted by t.
- Input-Output Sequence: It accepts a sequence of input vectors x(t) and generates a sequence of outputs y(t), where t represents the time step.

Network Connections:
1. Input to Hidden Layer:
   - Weight Matrix: Inputs x(t) are linked to the hidden layer via weight matrix W.
   - Activation Function: Neurons in the hidden layer apply activation function φ to the weighted sum of inputs plus bias vector b, yielding hidden state h(t).

2. Hidden Layer to Output:
   - Weight Vector: Hidden state h(t) is connected to the output neuron through weight vector v.
   - Previous Output Inclusion: For t > 1, the previous output y(t-1) contributes to the current state with weight r.
   - Activation and Bias: The output y(t) is computed using activation function φ on the weighted sum of hidden states and previous output (if applicable), with an added scalar c, or c_0 for t = 1.

3. Recurrent Connections:
   - Previous Output Dependency: Output y(t) incorporates the previous output y(t-1) multiplied by weight r.
   - Sequential Data Handling: The recurrent connection characterized by weight r enables the network to retain information across time steps, essential for processing sequential data.
```

We want to design this RNN in such a way that it determines whether the two binary input sequences x are identical or not upon receipt.

In this network, the calculations performed at each time step t are as follows:

$$h(t) = \phi(W x(t) + b)$$
$$y(t) =
\begin{cases}
\phi(v^T h(t) + ry^{t-1} + c) & \text{if } t > 1 \\
\phi(v^T h(t) + c_0) & \text{if } t = 1
\end{cases}$$

where $\phi$ is the step activation function, $W$ is a 2 × 2 matrix, $b$ and $v$ are 2-dimensional vectors, and $r$, $c$, and $c_0$ are scalars.

In this network, at each time step t, the elements $x_1(t)$ and $x_2(t)$ are given as input to the model, and the output $y(t)$ calculates whether all pairs of elements up to time t have matched or not. In this way, the output $y(N)$ will indicate the equality / inequality of the two binary input sequences.

Determine the parameters $W$, $b$, $v$, $r$, $c$, and $c_0$ in such a way that the given network implements the desired function.
