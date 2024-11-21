# dAiv Reinforcement Learning Basics

## Dependencies
To run this project, ensure Python 3.8 or later is installed. Follow the steps below to set up the environment:

### Install Required Packages
```bash
pip install -r requirements.txt
```

### Install PyTorch
For NVIDIA GPUs (CUDA versions):

CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CUDA 12.4
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

using Mac or Window cpu environment
```bash
pip install torch torchvision torchaudio
```

## How to Run
### 1. Play the Snake Game
Run the playable snake game:
```
python snake_playable.py
```

### 2. Evolutionary Snake Game
Use genetic algorithms to evolve the snake:
```
python genetic/main_genetic.py
```

### 3. Deep Q-Learning Snake Game
Run the DQN-based snake AI:
```
python dqn/main_dqn.py
```

## Credits
Snake game code by HonzaKral: https://gist.github.com/HonzaKral/833ee2b30231c53ec78e
