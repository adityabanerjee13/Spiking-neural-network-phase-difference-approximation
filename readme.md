# Phase Difference Prediction using Spiking Neural Networks (SNNs) and Artificial Neural Networks (ANNs)

## Introduction

In this project, I explore the use of **Spiking Neural Networks (SNNs)** to predict the **phase difference between two signals**, formulating the task as a **classification problem**. The objective is to compare the performance of SNNs against traditional **Artificial Neural Networks (ANNs)** of equivalent complexity.

### Spiking Neural Networks (SNNs)

Spiking Neural Networks (SNNs) are biologically inspired neural networks that simulate the behavior of neurons in the brain more closely than traditional ANNs. While ANNs process information in continuous activations, SNNs operate based on the timing of discrete events, known as **spikes**. Neurons in SNNs communicate via these spikes, only activating when their membrane potential exceeds a certain threshold, which makes them more energy efficient and better suited for time-dependent tasks such as signal processing, robotics, and neuromorphic computing.

SNNs can capture temporal dynamics naturally and are often used for tasks involving **time-series data** or tasks where timing is crucial. However, training SNNs is typically more challenging compared to ANNs due to the non-differentiable nature of spike-based communication, necessitating specialized algorithms like **spike-timing-dependent plasticity (STDP)**, **surrogate gradient descent**, or other techniques.

### Artificial Neural Networks (ANNs)

ANNs are widely used in deep learning and consist of layers of interconnected neurons, each transforming the input using a weighted sum and a non-linear activation function (e.g., ReLU, sigmoid). Unlike SNNs, ANNs are continuous and differentiable, which allows them to be trained using backpropagation. ANNs do not inherently model time, but they can handle time-series data when combined with architectures such as **recurrent neural networks (RNNs)** or **long short-term memory (LSTM)** cells, but in this experiment.

## Experiment Overview

In this experiment, I compare the performance of SNNs and ANNs for predicting the **phase difference** between two sinusoidal signals. This task is formulated as a **classification problem**, where the phase difference is discretized into 180 classes for every degree. Both the SNN and ANN models are designed to have **equivalent complexity** in terms of the number of neurons and layers to provide a fair comparison.

### Key Steps

1. **Data Generation**: 
   - Two sinusoidal signals are generated with varying phase differences.
   - The task is to classify the phase difference between these signals into predefined classes.
   
2. **Model Architecture**: 
   - **ANN Model**: A feedforward neural network with 2 linear layers and Sigmoid activation.
   - **SNN Model**: A spiking neural network with a similar architecture, but using spiking neurons (i.e., LIF neurons) instead of sigmoid.

3. **Training**:
   - Both models are trained on the same dataset.
   - The ANN is trained using traditional backpropagation.
   - The SNN is trained using surrogate gradient descent, which allows backpropagation through the spiking neurons.

4. **Evaluation**:
   - The performance of the models is evaluated on a test set, and one observation is grafically displayed.
   - Accuracy, but with a relaxation of epsilon distance from true class.

## Results and Findings

After conducting the experiments, the following observations were made:

- **Accuracy**: The ANN model tends to achieve lower classification accuracy by a large margin due to the simplicity of the model.

- **loss**: The ANN model does not decrese the classification(cross entropy) loss much with the lowest being 5.02 but the SNN achives a loss as low as 2.001.

- **Temporal Dynamics**: SNNs are better suited for capturing the temporal dynamics of signals. This is particularly useful when dealing with time-varying data which it is the case here.

- **forward path Complexity**: One forward path complexity in SNNs is more than that of ANN because of the inherent temporal computation in SNN and the memory state(membrane potential).