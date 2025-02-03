# Rock, Paper, Scissors Using EMG Gesture Recognition

This project implements a gesture-controlled **Rock, Paper, Scissors** game using **electromyographic (EMG) recordings**. The system recognizes hand gestures using a machine learning model trained on EMG data, allowing players to interact with the game by making the gestures for rock, paper, or scissors.

## Features
- **Gesture Recognition:** Implements **Multi-Layer Perceptron (MLP), Neural Network (NN), and Long Short-Term Memory (LSTM)** to classify EMG signals into four categories (rock, paper, scissors, and ok).
- **Gameplay Simulation:** Players select a gesture, and the model recognizes and plays against them.
- **Adaptive AI Opponent:** Uses **Temporal Difference (TD) Learning** to improve decision-making over time.

## Dataset
The project uses the **Classify gestures by reading muscle activity** dataset from Kaggle:
[Classify gestures by reading muscle activity](https://www.kaggle.com/datasets/kyr7plus/emg-4)

- Data recorded using a **MYO armband** with 8 EMG sensors.
- Each sample contains **64 EMG readings** over **40 ms** at **200 Hz**.
- Four classes:
  - `0` - Rock
  - `1` - Paper
  - `2` - Scissors
  - `3` - Ok (used to start/end the game)

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
python3 -m pip install numpy pandas scikit-learn tensorflow torch matplotlib
```

### Clone the Repository
```bash
git clone https://github.com/your-username/rock-paper-scissors-gesture.git
cd rock-paper-scissors-gesture
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Run the Program
Use the following command to train the model and start the game (the parameters should be set accordingly):
```bash
python main.py --model_name LSTM --num_epochs 30 --batch_size 128 --results_path ../results --exp_name test --alpha 0.5 --gamma 0.1 --epsilon 0.5
```

- If a trained model is found, it will be loaded.
- If no trained model exists, it will train a new model and save it.
- After training or loading the model, the **Rock, Paper, Scissors** game starts automatically.

## Model Performance
### Evaluation Metrics
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------|------------|------------|------------|------------|------------|
| MLP | 93.28% | 93.38% | 93.38% | 93.38% | 98.92% |
| NN | 93.49% | 93.53% | 93.49% | 93.50% | 99.06% |
| LSTM | **97.47%** | **97.48%** | **97.47%** | **97.48%** | **99.80%** |

The **LSTM model** demonstrated the highest accuracy, making it the most effective choice for gesture recognition.

## AI Opponent Strategy
The AI opponent uses **Temporal Difference (TD) Learning**:
- Evaluates moves based on **Q-values**.
- Balances **exploration (random moves) and exploitation (best-known moves)**.
- Updates strategy **over multiple rounds** to improve its performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




