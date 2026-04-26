# CS224W Colab 4: Graph Attention Networks on Cora

## Overview

This project is based on **CS224W Colab 4** and focuses on implementing and training a **Graph Attention Network (GAT)** for node classification on the **Cora citation network**.

The notebook builds on earlier graph neural network concepts, including GCN and GraphSAGE, and introduces the attention-based GNN architecture proposed in the paper:

> Graph Attention Networks, Veličković et al. (2018)

The main objective is to implement a custom GAT message-passing layer using **PyTorch Geometric**, train it on the Cora dataset, and evaluate its node classification accuracy.

## Project Goals

The goals of this project are to:

1. Understand how message passing works in graph neural networks.
2. Implement a custom GAT layer using PyTorch Geometric’s `MessagePassing` class.
3. Apply multi-head attention to graph-structured data.
4. Train a GNN model on the Cora citation network.
5. Evaluate the model using node classification accuracy.
6. Visualize learned node embeddings using t-SNE.

## Dataset

This project uses the **Cora dataset**, a standard benchmark dataset for node classification in graph machine learning.

In the Cora graph:

- Nodes represent academic papers.
- Edges represent citation links between papers.
- Node features are bag-of-words representations of paper content.
- Each node belongs to one of several research topic classes.

Dataset statistics:

- **2,708 nodes**
- **5,429 edges**
- **1,433 node features**
- **7 prediction classes**

## Main Concepts

### Graph Neural Networks

Graph Neural Networks learn node, edge, or graph representations by repeatedly passing messages between neighboring nodes.

Each node updates its representation by combining information from its neighbors.

### Graph Attention Networks

Graph Attention Networks improve standard message passing by learning how much attention each node should pay to each of its neighbors.

Instead of treating all neighbors equally, GAT assigns learnable attention weights to neighboring nodes.

This allows the model to focus on the most important neighbors during aggregation.

### Multi-Head Attention

The GAT layer uses multiple attention heads to stabilize learning and capture different types of neighborhood information.

Each attention head learns a different representation, and the outputs are combined to form the final node embedding.

## Notebook Structure

The notebook is organized into the following sections:

### 1. Device Setup

The notebook recommends using a GPU runtime for faster training.

### 2. Installation

The notebook installs the required PyTorch and PyTorch Geometric packages, including:

- `torch`
- `torch-geometric`
- `torch-scatter`
- `torch-sparse`
- `ogb`

### 3. GNN Stack Module

A general `GNNStack` class is provided. This class allows different GNN layers, such as GraphSAGE or GAT, to be plugged into the same training pipeline.

### 4. Custom Message Passing Layer

The project introduces how to build a custom PyTorch Geometric message passing layer.

The key methods involved are:

- `forward`
- `message`
- `aggregate`

### 5. GAT Implementation

The main implementation task is the custom `GAT` class.

The GAT layer includes:

- Linear transformations for node features
- Learnable attention parameters
- Multi-head attention
- Attention score computation
- Softmax normalization over neighbors
- Dropout on attention coefficients
- Message aggregation

### 6. Optimizer Setup

The notebook provides a helper function to build optimizers.

The default optimizer is Adam.

### 7. Training and Testing

Training and evaluation functions are provided.

The model is trained on the Cora dataset and evaluated using test accuracy.

### 8. Embedding Visualization

The notebook includes a helper function that uses t-SNE to visualize learned node embeddings in two dimensions.

## Requirements

To run this project, you need:

- Python 3
- Google Colab or Jupyter Notebook
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn
- matplotlib
- NetworkX
- tqdm

The notebook includes installation commands for the main dependencies.

## How to Run

1. Open the notebook in Google Colab.
2. Set the runtime to GPU:

   ```text
   Runtime → Change runtime type → Hardware accelerator → GPU
