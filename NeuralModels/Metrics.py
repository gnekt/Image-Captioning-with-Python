import pandas as pd

class Result():
    
    def __init__(self):
        self.train_results = pd.DataFrame([], columns=["Epoch", "IDBatch", "Loss", "Accuracy"])
        convert_dict = {'Epoch': int,
                        "IDBatch": int,
                        "Loss":float,
                        'Accuracy': float
               }
        self.train_results = self.train_results.astype(convert_dict)
        
        convert_dict = {'Epoch': int,
                        'Accuracy': float
               }
        self.validation_results = pd.DataFrame([], columns=["Epoch", "Accuracy"])
        self.validation_results = self.validation_results.astype(convert_dict)
        
    def add_train_info(self, epoch: int, batch_id: int, loss: float, accuracy: float):
        self.train_results = self.train_results.append({"Epoch":epoch,"IDBatch": batch_id, "Loss": loss, "Accuracy":accuracy}, ignore_index=True)
        
    def add_validation_info(self, epoch: int, accuracy: float):
        self.validation_results = self.validation_results.append({"Epoch":epoch, "Accuracy":accuracy}, ignore_index=True)
        
    def flush(self, directory: str = "."): 
        self.train_results.to_csv(f'{directory}/train_results.csv', encoding='utf-8', index=False)
        self.validation_results.to_csv(f'{directory}/validation_results.csv', encoding='utf-8', index=False)