from transformers import BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
import wandb
wandb.login()

class Finetuner():
    def __init__(self,data: pd.DataFrame, model: str, metric: str, training_arguments:dict, path_to_save:str) -> None:
        """We initialize sequentially all the prerequisites for the fine tuning
        Args:
            data (pd.DataFrame): preprocessed data with which we will fine tune our model
            model (str): The model chosen to performe fine tuning on
            metric (str): the metric chose to evaluate our model
            training_arguments (dict): hyperparameters of our model
            path_to_save (str): saving path
        """
        self.model = self.init_model(model)
        self.tokenizer = self.init_tokenizer(model)
        self.tokenized_datasets = self.init_dataset(data)
        self.metric = load_metric(metric)
        self.training_arguments = self.init_training_arguments(training_arguments)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = self.init_trainer()
        self.path_to_save = path_to_save

    def init_tokenizer(self, model:str):
        return BertTokenizer.from_pretrained(model)
    
    def init_model(self, model:str):
        return AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
    
    def init_dataset(self, df:pd.DataFrame):
        """We transform our pandas dataframe into a readable dataset for our model.
        - By splitting our dataset into test and train
        - By Tokenizing our inputs

        Args:
            df (pd.DataFrame): preprocessed data

        Returns:
            HF dataset: split and tokenized dataset
        """
        dataset = Dataset.from_pandas(df[['Text_lower', 'Label']])
        dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
        dataset = dataset.rename_columns({'Text_lower': 'text', 'Label':'label'})
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        return tokenized_datasets
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def init_training_arguments(self, training_arguments:dict):
        # Please note that I used a M1 mac for the finetuning. You migh need to modify this argument in the config: use_mps_device
        return TrainingArguments(**training_arguments)
    
    def compute_metric(self, eval_predictions):
        predictions, labels = eval_predictions
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def init_trainer(self):
        """we generate a trainer with our configuration and the already initialized components
        Returns:
            Trainer: trainer that wil perform fine tuning on top of pre-saved checkpoints
        """
        trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metric
        )
        return trainer
    
    def train(self):
        self.trainer.train()
        self.save_model()

    def save_model(self):
        self.trainer.save_model(self.path_to_save)
        print(f'Model finetuned and saved in {self.path_to_save}')

    def evaluate(self):
        self.trainer.evaluate()

    