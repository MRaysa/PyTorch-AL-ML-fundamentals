# PyTorch AI & ML Fundamentals

A comprehensive collection of implementations covering fundamental to advanced deep learning concepts using PyTorch. This repository documents a complete learning journey from basic perceptrons to modern transformer architectures.

## 📚 Contents

### 1. Foundations
- **Linear Classification** (`1 (linear classification).ipynb`)
  - Basic perceptron implementation
  - Binary classification from scratch
  - Visualization of decision boundaries

- **Tensors** (`2 (tensor).ipynb`)
  - PyTorch tensor operations
  - Tensor manipulation and reshaping
  - Matrix operations fundamentals

- **Perceptron in PyTorch** (`3 (Perceptron_in_torch).ipynb`)
  - Building perceptrons with PyTorch
  - Forward and backward propagation
  - Weight updates and training loops

### 2. Regression
- **Linear Regression** (`4 (linear_regression).ipynb`)
  - From scratch implementation
  - Manual weight updates
  - Loss calculation

- **Linear Regression with Gradient Descent** (`5 (linear_regression_with_gradient).ipynb`)
  - Automatic differentiation with PyTorch
  - Gradient-based optimization
  - Learning rate exploration

- **Unified Regression & Classification** (`6(linear_regression_and_classifier_with_gradient).ipynb`)
  - Using `torch.nn.Linear`
  - SGD optimizer implementation
  - MSE and BCE loss functions

### 3. Classification
- **Linear Classifier in PyTorch** (`7 (Linear Classifier in torch)`)
  - Binary classification with neural networks
  - Sigmoid activation
  - BCEWithLogitsLoss

### 4. Neural Networks
- **Multi-Layer Perceptron for XOR** (`8 (XOR_mlp).ipynb`)
  - Solving non-linearly separable problems
  - Hidden layer architecture
  - ReLU activation functions

- **MLP for MNIST** (`9 (mlp_mnist).ipynb`)
  - Multi-class classification
  - MNIST digit recognition
  - CrossEntropyLoss
  - Model evaluation and accuracy

### 5. Convolutional Neural Networks
- **CNN with Transfer Learning** (`cnn_cifar10_transfer.ipynb`)
  - MobileNetV2 pre-trained model
  - CIFAR-10 classification
  - Fine-tuning strategies
  - Data augmentation

- **CNN for KIE** (`cnn_kie.ipynb`)
  - Custom CNN architecture
  - Advanced image processing

### 6. Transformers
- **Next Word Prediction** (`transformer_next_word.py`)
  - Transformer encoder architecture
  - Positional embeddings
  - Self-attention mechanism
  - Language modeling

- **Sequence-to-Sequence Prompting** (`transformer_seqr_prompt.py`)
  - Advanced transformer applications
  - Prompt engineering
  - Text generation

## 🎓 Learning Resources

The repository includes comprehensive PDF materials covering:
- AI & ML Introduction
- ML Classifiers
- Gradient Descent
- Perceptrons
- Activation Functions
- Loss Functions
- XOR Problem
- Tensors

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision torchsummary numpy matplotlib pandas
```

### Running the Notebooks
1. Clone the repository:
```bash
git clone https://github.com/MRaysa/pytorch-AL-ML-fundamentals.git
cd pytorch-AL-ML-fundamentals
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open any notebook and run the cells sequentially

### Running Python Scripts
```bash
python transformer_next_word.py
python transformer_seqr_prompt.py
```

## 📊 Topics Covered

- ✅ Linear Algebra with Tensors
- ✅ Perceptron Algorithm
- ✅ Linear Regression
- ✅ Logistic Regression
- ✅ Gradient Descent Optimization
- ✅ Backpropagation
- ✅ Multi-Layer Perceptrons
- ✅ Activation Functions (ReLU, Sigmoid)
- ✅ Loss Functions (MSE, BCE, CrossEntropy)
- ✅ Convolutional Neural Networks
- ✅ Transfer Learning
- ✅ Transformer Architecture
- ✅ Self-Attention Mechanism

## 🛠️ Technologies

- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision datasets and models
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization

## 📈 Project Structure

```
pytorch-AL-ML-fundamentals/
├── 1 (linear classification).ipynb
├── 2 (tensor).ipynb
├── 3 (Perceptron_in_torch).ipynb
├── 4 (linear_regression).ipynb
├── 5 (linear_regression_with_gradient).ipynb
├── 6(linear_regression_and_classifier_with_gradient).ipynb
├── 7 (Linear Classifier in torch)
├── 8 (XOR_mlp).ipynb
├── 9 (mlp_mnist).ipynb
├── cnn_cifar10_transfer.ipynb
├── cnn_kie.ipynb
├── transformer_next_word.py
├── transformer_seqr_prompt.py
├── *.pdf (Learning materials)
└── README.md
```

## 🎯 Learning Path

**Beginner** → Linear Classification, Tensors, Basic Perceptron  
**Intermediate** → Regression, MLP, XOR Problem  
**Advanced** → CNNs, Transfer Learning, Transformers

## 💡 Key Concepts Demonstrated

1. **Building from Scratch**: Understanding algorithms by implementing them manually
2. **PyTorch Progression**: Gradual transition from manual implementations to PyTorch modules
3. **Real Datasets**: Working with MNIST, CIFAR-10, and custom datasets
4. **Modern Architectures**: From perceptrons to state-of-the-art transformers
5. **Best Practices**: Proper train/test splits, validation, and model evaluation

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**MRaysa**

---

⭐ Star this repository if you find it helpful for your deep learning journey!
