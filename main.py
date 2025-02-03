# Imports
import os
import torch
import argparse
from dataset import get_loaders
from learning_approach import LearningApproach
from model import get_model
from rock_paper_scissors import RockPaperScissors

def main(args):
    trn_loader, val_loader, tst_loader = get_loaders(args.batch_size)

    model_path = os.path.join(args.model_path, 'trained_model.pth')
    model = get_model(args.model_name)
    appr = LearningApproach(args.model_name, model, args.num_epochs)

    if args.retrain is False and os.path.exists(model_path):
        print('Loading saved trained model from {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))
    else:
        print('No saved model found, starting training from scratch')
        appr.train(trn_loader, val_loader)

        # Save the model after training
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print('Model saved to {}'.format(model_path))

    # Evaluation
    appr.eval(tst_loader)
    # Save test results
    print('Save at ' + os.path.join(args.model_path))

    # Rock, Paper, Scissors
    RockPaperScissors(model, args.alpha, args.gamma, args.epsilon).game()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example Python Framework")

    # Training arguments
    parser.add_argument("--model_path", type=str, default='../gesture-recognition-model', help="Path for saved model")
    parser.add_argument("--retrain", type=str, default=True, help="Retrain saved model or use saved model")
    parser.add_argument("--model_name", type=str, default='LSTM', help="Name of the model to use")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--batch_size", type=float, default=128, help="Batch size for training")
    parser.add_argument("--num_epochs", type=float, default=30, help="Number of epochs for training")

    # Game arguments
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Exploration rate")

    args = parser.parse_args()
    main(args)