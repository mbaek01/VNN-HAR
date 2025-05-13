import os
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from models.model import Model
from train.rotation import rotation
from utils import EarlyStopping, adjust_learning_rate_class

class Trainer:
    def __init__(self, args, model, curr_save_path):
        self.model = model.model
        self.device = model.device
        self.train_rot = args.train_rot

        self.criterion = model.select_criterion().to(self.device)
        self.optimizer = model.select_optimizer()

        self.curr_save_path = curr_save_path
        self.epoch_log_file_name = os.path.join(self.curr_save_path, "epoch_log.txt")
        
        self.epoch_log = open(self.epoch_log_file_name, "a")
        self.epochs = args.train_epochs

        self.early_stopping = EarlyStopping(metric="f1_macro", patience=args.early_stop_patience, verbose=True)
        # self.learning_rate_adapter = adjust_learning_rate_class(args, True)

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = []
        epoch_time = time.time()
        
        for batch_x1, batch_y in train_loader:
            batch_x1 = batch_x1.double().to(self.device)
            batch_y = batch_y.long().to(self.device)

            # Applying rotation to the data
            if self.train_rot in ["so3", "z"]:
                batch_x1 = rotation(batch_x1, self.train_rot, self.device)

            outputs = self.model(batch_x1)
            loss = self.criterion(outputs, batch_y)

            train_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_time = time.time() - epoch_time
        train_loss = np.average(train_loss)
        
        return train_loss, epoch_time

    def validate(self, valid_loader):
        self.model.eval()
        total_loss = []
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch_x1, batch_y in valid_loader:

                batch_x1 = batch_x1.double().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # Applying rotation to the data
                if self.train_rot in ["so3", "z"]:
                    batch_x1 = rotation(batch_x1, self.train_rot, self.device)


                outputs = self.model(batch_x1)
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())

                pred = outputs.detach().argmax(dim=1).cpu().numpy() # argmax?
                true = batch_y.detach().cpu().numpy()
                
                predictions.extend(pred)
                true_labels.extend(true)
                
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        valid_loss = np.average(total_loss)

        acc = accuracy_score(true_labels, predictions)
        f_w = f1_score(true_labels, predictions, average='weighted')
        f_macro = f1_score(true_labels, predictions, average='macro')
        # f_micro = f1_score(true_labels, predictions, average='micro')

        return valid_loss, acc, f_w, f_macro, # f_micro

    def train(self, train_loader, valid_loader):
        print("Epoch Log File: ", self.epoch_log_file_name)
    
        for epoch in range(self.epochs):
            train_loss, epoch_time = self.train_epoch(train_loader)
            
            if self.epoch_log is not None:
                self.epoch_log.write(f"Epoch: {epoch+1}, train_loss: {train_loss}, Cost time: {epoch_time}\n")
                
            print(f"Epoch: {epoch+1}, train_loss: {train_loss}, Cost time: {epoch_time}")

            # Validation phase
            valid_loss, valid_acc, valid_f_w, valid_f_macro = self.validate(valid_loader)
            
            print(f"VALID: \n Epoch: {epoch+1}, "
                  f"Train Loss: {train_loss:.7f}, "
                  f"Valid Loss: {valid_loss:.7f}, "
                  f"Valid Accuracy: {valid_acc:.7f}, "
                #   f"Valid weighted F1: {valid_f_w:.7f}, "
                  f"Valid macro F1: {valid_f_macro:.7f}, "
                #   f"Valid micro F1: {valid_f_micro:.7f}"
                )
        
            self.epoch_log.write(f"VALID: Epoch: {epoch+1}, "
                                f"Train Loss: {train_loss:.7f}, "
                                f"Valid Loss: {valid_loss:.7f}, \n"
                                f"Valid Accuracy: {valid_acc:.7f}, "
                                # f"Valid weighted F1: {valid_f_w:.7f}, "
                                f"Valid macro F1: {valid_f_macro:.7f}, \n"
                                # f"Valid micro F1: {valid_f_micro:.7f} \n"
                                )

            # Early stopping
            self.early_stopping(valid_loss, self.model, self.curr_save_path, valid_f_macro, valid_f_w, self.epoch_log)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            self.epoch_log.write("----------------------------------------------------------------------------------------\n")
            self.epoch_log.flush()
            
            # Learning rate adapter
            # self.learning_rate_adapter(self.optimizer, valid_loss)
        return self.model 


def test_predictions(args, test_loader, curr_save_path, score_log, test_sub):
    # Load model 
    model_path = os.path.join(curr_save_path, "best_vali.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model: '{model_path}' does not exist.")

    model = Model(args)
    device = model.device
    model = model.model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds = []
    trues = []

    # Testing phase
    for i, (batch_x1, batch_y) in enumerate(test_loader):
        batch_x1 = batch_x1.double().to(device)
        batch_y = batch_y.long().to(device)

        # Applying rotation to the data
        if args.test_rot in ["so3", "z"]:
            batch_x1 = rotation(batch_x1, args.test_rot, device)

        outputs = model(batch_x1)
    
        preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
        trues.extend(list(batch_y.detach().cpu().numpy()))
    
    acc = accuracy_score(preds,trues)
    # f_w = f1_score(trues, preds, average='weighted')
    f_macro = f1_score(trues, preds, average='macro')
    # f_micro = f1_score(trues, preds, average='micro')

    metrics_str = (
        f"Model: {model_path} | test subject: {test_sub} \n"
        f"Accuracy: {acc:.7f} | "
        # f"F1 Weighted: {f_w:.7f} | "
        f"F1 Macro: {f_macro:.7f} | "
        # f"F1 Micro: {f_micro:.7f} \n"
    )
    print(metrics_str)

    # Test log
    score_log.write(metrics_str)
    score_log.write("----------------------------------------------------------------------------------------\n")
    score_log.flush()

    print("Test Complete!")

    return acc, f_macro