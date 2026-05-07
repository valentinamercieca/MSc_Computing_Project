#SPACY NER TRAINING AND HYPERPARAMETER SEARCH
#Training a spaCy NER model directly from the final Label Studio JSON export
#using character-level entity offsets.

#The best model is selected based on development set F1 score and then
#evaluated on a held-out test set.

#Outputs:
# - Best trained spaCy model
# - CSV file with hyperparameter search results
# - Final evaluation metrics on the test set
#-----------------------------------------------------------------------------

import json
import random
import csv
import shutil
import time
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch

#Setting seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

VALID_LABELS = {
    "PROPERTY_NUMBER",
    "PROPERTY_NAME",
    "SUB_UNIT",
    "STREET_PREFIX",
    "STREET_NAME",
    "LOCALITY",
    "ISLAND",
}

#Input and output paths
BASE_DIR = Path(r"D:\ML_Experiments\Final\Outputs")
INPUT_JSON = BASE_DIR / "Annotation_3000.json"

RESULTS_CSV = BASE_DIR / "hyperparameter_results.csv"
BEST_MODEL_DIR = BASE_DIR / "spacy_model_best"
TEMP_BEST_RUN_DIR = BASE_DIR / "temp_best_run_model"

#Hyperparameter grid
EPOCH_VALUES = [15, 25, 35]
BATCH_VALUES = [4, 8, 16]
DROPOUT_VALUES = [0.1, 0.2, 0.3]

#Loading final Label Studio JSON export
def load_tasks(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#Splitting tasks into train/dev/test sets
def split_tasks(tasks):
    tasks = tasks.copy()
    random.shuffle(tasks)
    train = tasks[:2100]
    dev = tasks[2100:2550]
    test = tasks[2550:]
    return train, dev, test

#Converting one Label Studio task into a spaCy training format
def task_to_spacy_example(task):
    text = task["data"]["raw_address_string"]
    annotations = task.get("annotations", [])
    if not annotations:
        return None

    result = annotations[0].get("result", [])
    entities = []

    for item in result:
        if item.get("type") != "labels":
            continue

        value = item.get("value", {})
        start = value.get("start")
        end = value.get("end")
        labels = value.get("labels", [])

        if start is None or end is None or not labels:
            continue

        label = labels[0]

        #Retain only labels defined in the annotation scheme
        if label not in VALID_LABELS:
            continue

        entities.append((start, end, label))

    #Sort spans for consistent processing
    entities = sorted(entities, key=lambda x: (x[0], x[1]))
    return (text, {"entities": entities})

#Building spaCy Example objects
def build_examples(nlp, tasks):
    examples = []
    skipped = 0

    for task in tasks:
        converted = task_to_spacy_example(task)
        if converted is None:
            continue

        text, annotations = converted
        doc = nlp.make_doc(text)

        try:
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        except Exception:
            #Skip cases with incompatible spans
            skipped += 1

    return examples, skipped

#Evaluate model using spaCy's built-in scorer
def evaluate_model(nlp, examples):
    return nlp.evaluate(examples)

#Print entity-level precision, recall, and F1
def print_per_entity_scores(scores):
    print("\n=== Per-Entity Results ===")
    ents_per_type = scores.get("ents_per_type", {})

    for label, metrics in ents_per_type.items():
        p = metrics.get("p", 0.0)
        r = metrics.get("r", 0.0)
        f = metrics.get("f", 0.0)
        print(f"{label:16} P: {p:.4f} | R: {r:.4f} | F1: {f:.4f}")

#Create blank multilingual spaCy model with NER component
def create_model():
    nlp = spacy.blank("xx")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    #Register all entity labels before training
    for label in VALID_LABELS:
        ner.add_label(label)

    return nlp

#Replace destination model directory with source model directory
def copy_model_dir(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

#Training and evaluating one huperparameter configuration
def train_and_evaluate(train_tasks, dev_tasks, epochs, batch_size, dropout, temp_dir, seed=42):
    random.seed(seed)

    nlp = create_model()

    train_examples, skipped_train = build_examples(nlp, train_tasks)
    dev_examples, skipped_dev = build_examples(nlp, dev_tasks)

    print(f"Train examples: {len(train_examples)}, skipped: {skipped_train}")
    print(f"Dev examples:   {len(dev_examples)}, skipped: {skipped_dev}")

    optimizer = nlp.initialize(lambda: train_examples)

    best_dev_f1 = -1.0
    best_dev_scores = None
    best_epoch = -1

    epoch_times = []
    experiment_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        random.shuffle(train_examples)
        losses = {}

        #Train using mini-batches
        batches = minibatch(train_examples, size=batch_size)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses, drop=dropout)

        #Evaluate after each epoch
        dev_scores = evaluate_model(nlp, dev_examples)
        dev_f1 = dev_scores["ents_f"]

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"[epochs={epochs}, batch={batch_size}, dropout={dropout}] "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {losses.get('ner', 0):.4f} | "
            f"Dev P: {dev_scores['ents_p']:.4f} | "
            f"Dev R: {dev_scores['ents_r']:.4f} | "
            f"Dev F1: {dev_f1:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        #Save the best epoch model for this run
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_dev_scores = dev_scores
            best_epoch = epoch + 1

            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            nlp.to_disk(temp_dir)

    total_experiment_time = time.perf_counter() - experiment_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    return {
        "dev_p": best_dev_scores["ents_p"],
        "dev_r": best_dev_scores["ents_r"],
        "dev_f1": best_dev_scores["ents_f"],
        "best_epoch": best_epoch,
        "skipped_train": skipped_train,
        "skipped_dev": skipped_dev,
        "total_experiment_time_sec": total_experiment_time,
        "avg_epoch_time_sec": avg_epoch_time,
    }

#Evaluate saved best model on test set
def test_model(model_dir, test_tasks):
    test_start = time.perf_counter()

    nlp = spacy.load(model_dir)
    test_examples, skipped_test = build_examples(nlp, test_tasks)
    test_scores = evaluate_model(nlp, test_examples)

    test_time = time.perf_counter() - test_start

    return {
        "test_p": test_scores["ents_p"],
        "test_r": test_scores["ents_r"],
        "test_f1": test_scores["ents_f"],
        "skipped_test": skipped_test,
        "raw_scores": test_scores,
        "test_time_sec": test_time,
    }

#Main worflow
def main():
    overall_start = time.perf_counter()

    print("Loading tasks...")
    tasks = load_tasks(INPUT_JSON)
    print(f"Total tasks: {len(tasks)}")

    train_tasks, dev_tasks, test_tasks = split_tasks(tasks)
    print(f"Train: {len(train_tasks)}")
    print(f"Dev:   {len(dev_tasks)}")
    print(f"Test:  {len(test_tasks)}")

    #Remove previous model outputs
    if TEMP_BEST_RUN_DIR.exists():
        shutil.rmtree(TEMP_BEST_RUN_DIR)
    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)

    results = []
    best_overall = None
    best_dev_f1 = -1.0

    #Run full grid search
    for epochs in EPOCH_VALUES:
        for batch_size in BATCH_VALUES:
            for dropout in DROPOUT_VALUES:
                print("\n" + "=" * 80)
                print(
                    f"Running experiment: epochs={epochs}, "
                    f"batch_size={batch_size}, dropout={dropout}"
                )

                result = train_and_evaluate(
                    train_tasks=train_tasks,
                    dev_tasks=dev_tasks,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                    temp_dir=TEMP_BEST_RUN_DIR,
                    seed=RANDOM_SEED,
                )

                #Store development set results for this configuration
                row = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "best_epoch": result["best_epoch"],
                    "dev_precision": round(result["dev_p"], 6),
                    "dev_recall": round(result["dev_r"], 6),
                    "dev_f1": round(result["dev_f1"], 6),
                    "skipped_train": result["skipped_train"],
                    "skipped_dev": result["skipped_dev"],
                    "total_experiment_time_sec": round(result["total_experiment_time_sec"], 2),
                    "avg_epoch_time_sec": round(result["avg_epoch_time_sec"], 2),
                }
                results.append(row)

                #Track best overall model based on dev F1
                if result["dev_f1"] > best_dev_f1:
                    best_dev_f1 = result["dev_f1"]
                    best_overall = {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "dropout": dropout,
                        "best_epoch": result["best_epoch"],
                        "total_experiment_time_sec": result["total_experiment_time_sec"],
                        "avg_epoch_time_sec": result["avg_epoch_time_sec"],
                    }
                    copy_model_dir(TEMP_BEST_RUN_DIR, BEST_MODEL_DIR)

    #Save all dev set search results
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epochs",
                "batch_size",
                "dropout",
                "best_epoch",
                "dev_precision",
                "dev_recall",
                "dev_f1",
                "skipped_train",
                "skipped_dev",
                "total_experiment_time_sec",
                "avg_epoch_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved hyperparameter search results to:")
    print(RESULTS_CSV)

    #Report best development set configuration
    print("\n=== Best Hyperparameters Based on Dev F1 ===")
    print(f"Epochs:     {best_overall['epochs']}")
    print(f"Batch size: {best_overall['batch_size']}")
    print(f"Dropout:    {best_overall['dropout']}")
    print(f"Best epoch: {best_overall['best_epoch']}")
    print(f"Best Dev F1: {best_dev_f1:.4f}")
    print(f"Best experiment total time: {best_overall['total_experiment_time_sec']:.2f} seconds")
    print(f"Best experiment average epoch time: {best_overall['avg_epoch_time_sec']:.2f} seconds")

    #Final evaluation on unseen test data
    print("\nFinal evaluation on test set using saved best model...")
    test_result = test_model(BEST_MODEL_DIR, test_tasks)

    print("\n=== Test Results ===")
    print(f"Precision: {test_result['test_p']:.4f}")
    print(f"Recall:    {test_result['test_r']:.4f}")
    print(f"F1 score:  {test_result['test_f1']:.4f}")
    print(f"Test evaluation time: {test_result['test_time_sec']:.2f} seconds")

    print_per_entity_scores(test_result["raw_scores"])

    overall_time = time.perf_counter() - overall_start
    print(f"\nTotal search + final evaluation time: {overall_time:.2f} seconds")

    print(f"\nSaved best model to: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()