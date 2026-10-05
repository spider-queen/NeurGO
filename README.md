# NeurGO

**NeurGO: A Generative Meta-Black-Box Optimizer for Low-Budget Expensive Optimization**

> **Abstract:** Expensive black-box optimization is ubiquitous in science and engineering. To overcome the limitations of traditional surrogate-assisted methods, we propose **NeurGO**, a generative framework that directly synthesizes elite candidates from historical population states. By leveraging a Rank-Fitness Dual-Aware Attention mechanism and a Generative Module, NeurGO efficiently navigates the search space with a limited evaluation budget.

## 🚀 Overview

This repository provides the official PyTorch implementation of **NeurGO**. 

**Key Features:**
- **Generative Meta-Learning:** Directly generates high-quality candidates instead of screening large pools.
- **Dual-Aware Attention:** Captures both rank-based and fitness-based population information.
- **Comprehensive Evaluation:** Trained on Low-Fidelity (LF) tasks and tested on CEC Functions and the BBOB suite.

## 🛠️ Requirements

- **Python 3.10+** (Recommended)
- **PyTorch >= 1.12**

- **OS:** Ubuntu 22.04.5 LTS 

To set up the environment, please install the core dependencies first:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> We recommend installing PyTorch with CUDA support compatible with your specific hardware. Visit [pytorch.org](https://pytorch.org) for details.

## 📦 Installation
This project is organized into two local packages: NeurGO (core algorithm) and Benchmark (test functions). You must install them in editable mode to resolve imports correctly.

1. Install the NeurGO algorithm:
```bash
cd NeurGO_pkg
pip install -e .
```

2. Install the Benchmark suite:
```bash
cd Benchmark_pkg
pip install -e .
```

## 🏃‍♂️ Usage
We provide shell scripts for easy reproduction of the experiments.

**Training**

Train the NeurGO model on the Low-Fidelity (LF) tasks (LF1-LF3).

* Using Script:

```bash
chmod +x train.sh
bash train.sh
```

* Manual Command:

```bash
python main.py -d 10 -expname neurgo_demo -popsize 100 -maxepoch 1000 -lr 0.001 -mode train
```

**Testing**

Evaluate the trained model on standard test functions.

* Using Script:

```bash
chmod +x test.sh
bash test.sh
```

* Manual Command:

1. Test on CEC Functions (F1-F6):

```bash
python main.py -d 10 -expname neurgo_demo -popsize 100 -mode test -target sys
```

2. Test on BBOB Suite:

```bash
python main.py -d 10 -expname neurgo_demo -popsize 100 -mode test -target bbob
```
