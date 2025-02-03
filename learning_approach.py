# Imports
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.optim import Adam

class LearningApproach:
    def __init__(self, model_name, model, num_epochs):
        self.model_name = model_name
        self.model = model
        self.num_epochs = num_epochs
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, trn_loader, val_loader):
        print(f"Training Model:")

        # Save losses and accuracies for comparison
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            for inputs, labels in trn_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
            epoch_loss = running_loss / len(trn_loader.dataset)
            epoch_accuracy = correct_predictions / total_predictions
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            # Validation
            self.model.eval()
            val_running_loss = 0.0
            val_correct_predictions = 0
            val_total_predictions = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct_predictions += (predicted == labels).sum().item()
                    val_total_predictions += labels.size(0)
            val_loss = val_running_loss / len(val_loader.dataset)
            val_accuracy = val_correct_predictions / val_total_predictions
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    def eval(self, tst_loader):
        test_metrics = {}

        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tst_loader:
                outputs = self.model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        test_accuracy = correct_predictions / total_predictions
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

        test_metrics[self.model_name] = {
            'Accuracy': test_accuracy,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        }
        print(f'{self.model_name} - Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}')