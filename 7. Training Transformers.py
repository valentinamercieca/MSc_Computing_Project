#TRANSFORMER HYPERPARAMETER EXPERIMENTS AND MODELLING
#Training one transformer-based token classification model at a time using the fixed BIO
#train/dev/test splits

#A grid search is performed over learning rate, batch size, number of epochs,
#and weight decay. The best configuration is selected using development set F1
#and then evaluated on the held-out test set.
#-----------------------------------------------------------------------------

from pathlib import Path
import json
import csv
import gc
import shutil
import time
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

#-----------------------------------------------------------------------------
#1. BASIC SETTINGS

#Transformer model selected for this run:
#MODEL_NAME = "MLRS/mBERTu"
#MODEL_NAME = "google-bert/bert-base-cased"
#MODEL_NAME = "google-bert/bert-base-multilingual-cased"
#MODEL_NAME = "FacebookAI/xlm-roberta-base"
MODEL_NAME = "MLRS/BERTu"

#Input and output paths
BASE_DIR = Path(r"D:\ML_Experiments\Final\Outputs")
TRAIN_FILE = BASE_DIR / "train.txt"
DEV_FILE = BASE_DIR / "dev.txt"
TEST_FILE = BASE_DIR / "test.txt"

#Separate experiment folder for each transformer model
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")
EXPERIMENT_DIR = BASE_DIR / f"hf_hpsearch_{SAFE_MODEL_NAME}"

RESULTS_CSV = EXPERIMENT_DIR / "hyperparameter_results.csv"
BEST_SUMMARY_JSON = EXPERIMENT_DIR / "best_result_summary.json"
BEST_MODEL_DIR = EXPERIMENT_DIR / "best_model"

#Reproducibility settings
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

#Maximum tokenised sequence length
MAX_LENGTH = 128

#Hyperparameter grid
LEARNING_RATES = [2e-5, 3e-5]
TRAIN_BATCH_SIZES = [4, 8]
NUM_EPOCHS_LIST = [5, 10]
WEIGHT_DECAYS = [0.01]


#-----------------------------------------------------------------------------
#2. READ BIO FILES

def load_bio_file(path: Path):
    records = []
    tokens = []
    tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            #Blank line marks the end of one address sequence
            if not line.strip():
                if tokens:
                    records.append({"tokens": tokens, "ner_tags": tags})
                    tokens = []
                    tags = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            tag = parts[-1]

            tokens.append(token)
            tags.append(tag)

    #Add final sequence if file does not end with a blank line
    if tokens:
        records.append({"tokens": tokens, "ner_tags": tags})

    return records


#Loading fixed train/dev/test splits
train_records = load_bio_file(TRAIN_FILE)
dev_records = load_bio_file(DEV_FILE)
test_records = load_bio_file(TEST_FILE)

print("Train sequences:", len(train_records))
print("Dev sequences:", len(dev_records))
print("Test sequences:", len(test_records))


#-----------------------------------------------------------------------------
#3. LABEL MAPS

#BIO labels are derived from the training set only
unique_labels = sorted({tag for row in train_records for tag in row["ner_tags"]})

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print("Labels:", unique_labels)

#-----------------------------------------------------------------------------
#4. DATASETS

#Converting lists into Hugging Face Dataset objects
train_dataset = Dataset.from_list(train_records)
dev_dataset = Dataset.from_list(dev_records)
test_dataset = Dataset.from_list(test_records)

#-----------------------------------------------------------------------------
#5. METRICS

#Computing entity-level precision, recall, and F1 using seqeval
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        current_preds = []
        current_labels = []

        for pred_id, label_id in zip(pred_seq, label_seq):

            #Ignore special tokens and subword positions
            if label_id == -100:
                continue

            current_preds.append(id2label[int(pred_id)])
            current_labels.append(id2label[int(label_id)])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


#Building final test metrics and detailed entity-level classification report
def build_detailed_report(pred_output):
    logits = pred_output.predictions
    labels = pred_output.label_ids
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        current_preds = []
        current_labels = []

        for pred_id, label_id in zip(pred_seq, label_seq):

            #Ignore positions excluded from loss and evaluation
            if label_id == -100:
                continue
            
            current_preds.append(id2label[int(pred_id)])
            current_labels.append(id2label[int(label_id)])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "classification_report": classification_report(
            true_labels, true_predictions, digits=4
        ),
    }


#-----------------------------------------------------------------------------
#6. TOKENISER AND TOKENISATION

#Loading pretrained tokeniser for the selected transformer model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

#Tokenising pre-split tokens and aligning BIO labels to first subword tokens
#Special tokens and non-initial subwords receive label -100 so they are ignored
#during training and evaluation
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
    )

    all_labels = []

    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[word_labels[word_id]])
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized

#Applying tokenisation and label alignment to all splits
train_tokenized = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=train_dataset.column_names,
)

dev_tokenized = dev_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dev_dataset.column_names,
)

test_tokenized = test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=test_dataset.column_names,
)

#Dynamic padding for token classification batches
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


#-----------------------------------------------------------------------------
#7. TIMING CALLBACK

#Trainer callback used to record epoch-level training time
class EpochTimingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = []
        self._epoch_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start is not None:
            elapsed = time.perf_counter() - self._epoch_start
            self.epoch_times.append(elapsed)
            print(f"Epoch time: {elapsed:.2f}s")

    def average_epoch_time(self):
        if not self.epoch_times:
            return 0.0
        return sum(self.epoch_times) / len(self.epoch_times)


#-----------------------------------------------------------------------------
#8. SEARCH PHASE WITH CHECKPOINTING

#Training and evaluating one hyperparameter configuration
def run_one_experiment(learning_rate, train_batch_size, num_epochs, weight_decay):
    run_name = f"lr_{learning_rate}_bs_{train_batch_size}_ep_{num_epochs}_wd_{weight_decay}"
    run_output_dir = EXPERIMENT_DIR / run_name

    #Remove old output from the same configuration
    if run_output_dir.exists():
        shutil.rmtree(run_output_dir)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    timing_callback = EpochTimingCallback()

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[timing_callback],
    )

    print("\n" + "=" * 80)
    print(
        f"SEARCH | model={MODEL_NAME} | lr={learning_rate} | "
        f"batch={train_batch_size} | epochs={num_epochs} | wd={weight_decay}"
    )

    experiment_start = time.perf_counter()
    trainer.train()
    experiment_time = time.perf_counter() - experiment_start

    #Evaluate best checkpoint on the development set
    dev_metrics = trainer.evaluate(eval_dataset=dev_tokenized)

    best_epoch = None
    best_eval_f1 = -1.0

    #Identify epoch with best logged development F1
    for log in trainer.state.log_history:
        if "eval_f1" in log and log["eval_f1"] > best_eval_f1:
            best_eval_f1 = log["eval_f1"]
            best_epoch = log.get("epoch")

    result = {
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "dev_precision": dev_metrics["eval_precision"],
        "dev_recall": dev_metrics["eval_recall"],
        "dev_f1": dev_metrics["eval_f1"],
        "total_experiment_time_sec": experiment_time,
        "avg_epoch_time_sec": timing_callback.average_epoch_time(),
    }

    #Release memory before next run
    del trainer
    del model
    gc.collect()

    return result


#-----------------------------------------------------------------------------
#9. FINAL TRAINING WITH CHECKPOINTING

#Retraining the selected best configuration and evaluating it on the test set
def train_final_best_model(best_config):
    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    timing_callback = EpochTimingCallback()

    training_args = TrainingArguments(
        output_dir=str(BEST_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=best_config["learning_rate"],
        per_device_train_batch_size=best_config["train_batch_size"],
        per_device_eval_batch_size=best_config["train_batch_size"],
        num_train_epochs=best_config["num_epochs"],
        weight_decay=best_config["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[timing_callback],
    )

    print("\n" + "#" * 80)
    print("FINAL TRAINING WITH BEST CONFIG")
    print(best_config)
    print("#" * 80)

    final_train_start = time.perf_counter()
    trainer.train()
    final_train_time = time.perf_counter() - final_train_start

    final_dev_metrics = trainer.evaluate(eval_dataset=dev_tokenized)

    best_epoch = None
    best_eval_f1 = -1.0

    #Record best epoch from final training run
    for log in trainer.state.log_history:
        if "eval_f1" in log and log["eval_f1"] > best_eval_f1:
            best_eval_f1 = log["eval_f1"]
            best_epoch = log.get("epoch")

    final_dev_metrics["best_epoch"] = best_epoch

    #Evaluate final model on held-out test set
    test_start = time.perf_counter()
    test_output = trainer.predict(test_tokenized)
    test_time = time.perf_counter() - test_start

    detailed_report = build_detailed_report(test_output)

    #Save final model and tokeniser
    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))

    return {
        "final_dev_metrics": final_dev_metrics,
        "final_test_metrics": test_output.metrics,
        "detailed_report": detailed_report,
        "final_training_time_sec": final_train_time,
        "final_avg_epoch_time_sec": timing_callback.average_epoch_time(),
        "test_time_sec": test_time,
    }


#-----------------------------------------------------------------------------
#10. MAIN WORKFLOW

#Running grid search, retaining best configuration, and evaluating on test data
def main():
    overall_start = time.perf_counter()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    best_config = None
    best_dev_f1 = -1.0

    #Run hyperparameter grid search
    for learning_rate in LEARNING_RATES:
        for train_batch_size in TRAIN_BATCH_SIZES:
            for num_epochs in NUM_EPOCHS_LIST:
                for weight_decay in WEIGHT_DECAYS:
                    row = run_one_experiment(
                        learning_rate=learning_rate,
                        train_batch_size=train_batch_size,
                        num_epochs=num_epochs,
                        weight_decay=weight_decay,
                    )
                    results.append(row)
                    
                    #Track best configuration by dev F1
                    if row["dev_f1"] > best_dev_f1:
                        best_dev_f1 = row["dev_f1"]
                        best_config = {
                            "learning_rate": learning_rate,
                            "train_batch_size": train_batch_size,
                            "num_epochs": num_epochs,
                            "weight_decay": weight_decay,
                            "best_epoch": row["best_epoch"],
                            "total_experiment_time_sec": row["total_experiment_time_sec"],
                            "avg_epoch_time_sec": row["avg_epoch_time_sec"],
                        }

    #Saving search results to CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "learning_rate",
                "train_batch_size",
                "num_epochs",
                "weight_decay",
                "best_epoch",
                "dev_precision",
                "dev_recall",
                "dev_f1",
                "total_experiment_time_sec",
                "avg_epoch_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    **row,
                    "total_experiment_time_sec": round(row["total_experiment_time_sec"], 2),
                    "avg_epoch_time_sec": round(row["avg_epoch_time_sec"], 2),
                }
                for row in results
            ]
        )

    print("\nBest hyperparameters from search:")
    print(best_config)
    print(f"Best dev F1: {best_dev_f1:.4f}")
    print(f"Best search best epoch: {best_config['best_epoch']}")
    print(f"Best search experiment total time: {best_config['total_experiment_time_sec']:.2f} seconds")
    print(f"Best search average epoch time: {best_config['avg_epoch_time_sec']:.2f} seconds")

    #Retrain and evaluate selected configuration
    final_result = train_final_best_model(best_config)

    summary = {
        "model_name": MODEL_NAME,
        "seed": RANDOM_SEED,
        "max_length": MAX_LENGTH,
        "best_hyperparameters": {
            "learning_rate": best_config["learning_rate"],
            "train_batch_size": best_config["train_batch_size"],
            "num_epochs": best_config["num_epochs"],
            "weight_decay": best_config["weight_decay"],
        },
        "search_best_dev_f1": best_dev_f1,
        "search_best_epoch": best_config["best_epoch"],
        "search_total_experiment_time_sec": round(best_config["total_experiment_time_sec"], 2),
        "search_avg_epoch_time_sec": round(best_config["avg_epoch_time_sec"], 2),
        "final_dev_metrics": final_result["final_dev_metrics"],
        "final_test_metrics": final_result["final_test_metrics"],
        "final_training_time_sec": round(final_result["final_training_time_sec"], 2),
        "final_avg_epoch_time_sec": round(final_result["final_avg_epoch_time_sec"], 2),
        "test_time_sec": round(final_result["test_time_sec"], 2),
        "detailed_test_report": final_result["detailed_report"]["classification_report"],
    }

    #Save final summary as JSON
    with open(BEST_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Precision: {final_result['detailed_report']['precision']:.4f}")
    print(f"Recall:    {final_result['detailed_report']['recall']:.4f}")
    print(f"F1:        {final_result['detailed_report']['f1']:.4f}")
    print(f"Final best epoch: {final_result['final_dev_metrics']['best_epoch']}")
    print(f"Final training time: {final_result['final_training_time_sec']:.2f} seconds")
    print(f"Final average epoch time: {final_result['final_avg_epoch_time_sec']:.2f} seconds")
    print(f"Test evaluation time: {final_result['test_time_sec']:.2f} seconds")

    overall_time = time.perf_counter() - overall_start
    print(f"Total script runtime: {overall_time:.2f} seconds")

    print(f"\nSaved search results to: {RESULTS_CSV}")
    print(f"Saved best summary to: {BEST_SUMMARY_JSON}")
    print(f"Saved best model to: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()