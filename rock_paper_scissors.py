import random
import torch
import pandas as pd
from dataset import get_data

class RockPaperScissors:
    def __init__(self, model, alpha, gamma, epsilon):
        # self.recordings_tst = recordings_tst
        # self.gestures_tst = gestures_tst
        self.model = model
        self.alpha = alpha # Exploration rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate

    def game(self):
        gesture_dict = {"OK": 0,
                        "Rock": 1,
                        "Paper": 2,
                        "Scissors": 3}

        # Game
        player_history = []
        computer_wins = 0
        player_wins = 0
        Q_values = {"Rock": 0, "Paper": 0, "Scissors": 0}

        while True:
            print("Type 'OK' to start the game!")
            start_game = input("Player input: ").strip().upper()
            if start_game == "OK":
                start_choice_sample = get_emg_sample(gesture_dict[start_game])
                start_prediction = predict_gesture(self.model, start_choice_sample)
                if start_prediction == "OK":
                    print("OK gesture recognized. Game started!")
                    break
                else:
                    print("OK gesture not recognized. Please try again.")
                break
            else:
                print("Invalid choice. Please type 'OK'.")

        if start_prediction == "OK":
            while True:
                player_choice = input(
                    "Choose your gesture (Rock, Paper, Scissors) or type 'OK' to end the game: ").capitalize()
                if player_choice.upper() == "OK":
                    end_choice_sample = get_emg_sample(gesture_dict[player_choice.upper()])
                    end_prediction = predict_gesture(self.model, end_choice_sample)
                    if end_prediction == "OK":
                        print("OK gesture recognized! Ending the game...")
                        break
                    else:
                        print("OK gesture not recognized. Please try again.")
                        continue

                elif player_choice not in gesture_dict:
                    print("Invalid choice. Please type Rock, Paper, Scissors, or OK.")
                    continue

                else:
                    choice_sample = get_emg_sample(gesture_dict[player_choice])
                    prediction = predict_gesture(self.model, choice_sample)

                    print(f"Player gesture recognized as: {prediction}")

                    # Computer makes a choice
                    if random.random() < self.epsilon:  # Exploration
                        computer_prediction = random.choice(["Rock", "Paper", "Scissors"])
                    else:  # Exploitation
                        computer_prediction = max(Q_values, key=Q_values.get)
                    print(f"Computer chose: {computer_prediction}")

                    # Determine winner and reward
                    result, reward = determine_winner(player_choice, computer_prediction)
                    print(f"Result: {result}!")

                    # Update scores
                    if result == "Player":
                        player_wins += 1
                    elif result == "Computer":
                        computer_wins += 1

                    # Update Q-values
                    # global Q_values
                    next_action = random.choice(list(Q_values.keys()))
                    next_Q = Q_values[next_action]
                    Q_values[player_choice] += self.alpha * (reward + self.gamma * next_Q - Q_values[player_choice])
                    print(f"Updated Q-values: {Q_values}\n")

                    # Track history
                    player_history.append(player_choice)

            # Game summary
            print("\nGame Over!")
            print(f"Total Rounds: {len(player_history)}")
            print(f"Player Wins: {player_wins}")
            print(f"Computer Wins: {computer_wins}")

            if player_wins > computer_wins:
                print("You are the overall winner :D")
            elif computer_wins > player_wins:
                print("The computer is the overall winner :(")
            else:
                print("It's a tie :|")

def sort_gestures():
    data = get_data()
    recordings_tst = pd.DataFrame(data['tst']['recordings']).reset_index(drop=True)
    gestures_tst = data['tst']['gestures'].reset_index(drop=True)
    tst_data_by_gesture = {}
    for gesture_label in gestures_tst.unique():
        indices = gestures_tst[gestures_tst == gesture_label].index
        tst_data_by_gesture[gesture_label] = recordings_tst.iloc[indices].values
    return tst_data_by_gesture

def get_emg_sample(gesture):
    # Get the samples for the selected gesture
    samples = sort_gestures()[gesture]
    # Randomly pick one sample
    selected_sample = random.choice(samples)
    return selected_sample

def predict_gesture(model, emg_sample):
    gesture_dict = {"OK": 0,
                    "Rock": 1,
                    "Paper": 2,
                    "Scissors": 3}
    emg_tensor = torch.tensor(emg_sample, dtype=torch.float32)
    # Pass the tensor through the model
    model.eval()
    with torch.no_grad():
        output = model(emg_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    # Map the predicted index back to the gesture label
    for gesture, label in gesture_dict.items():
        if label == predicted_class:
            return gesture

def determine_winner(player, computer):
    if player == computer:
        return "Draw", 0  # Reward 0 for draw
    elif (player == "Rock" and computer == "Scissors") or \
            (player == "Paper" and computer == "Rock") or \
            (player == "Scissors" and computer == "Paper"):
        return "Player", 1  # Reward +1 for player win
    else:
        return "Computer", -1  # Reward -1 for player loss