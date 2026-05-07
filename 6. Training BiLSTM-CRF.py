# BiLSTM-CRF Hyperparameter Search for Maltese Address NER

#Training a BiLSTM-CRF model using BIO files.

#The model is trained using fixed train, development, and test splits.
#Hyperparameter search is carried out in two phases:
#1. Tune epochs, learning rate, and batch size
#2. Tune embedding and hidden dimensions

#The best model is selected using development set F1 score and then evaluated
#on the held-out test set.

#-----------------------------------------------------------------------------

from pathlib import Path
from collections import Counter
import csv
import copy
import random
import time

import torch
import torch.nn as nn
from torchcrf import CRF

from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import f1_score

#Setting seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#Using CPU for training and evaluation
DEVICE = torch.device("cpu")

#Input and output paths
BASE = Path(r"D:\ML_Experiments\Final\Outputs")

TRAIN_FILE = BASE / "train.txt"
DEV_FILE = BASE / "dev.txt"
TEST_FILE = BASE / "test.txt"

RESULTS_CSV = BASE / "bilstm_crf_hyperparameter_results.csv"
BEST_MODEL_FILE = BASE / "bilstm_crf_best.pt"

#Phase 1 uses fixed model dimensions while tuning training settings
PHASE1_EMBED_DIM = 100
PHASE1_HIDDEN_DIM = 128

PHASE1_EPOCHS = [10, 15, 20]
PHASE1_LEARNING_RATES = [0.001, 0.0005]
PHASE1_BATCH_SIZES = [8, 16]

#Phase 2 tunes model capacity using the best Phase 1 training settings
PHASE2_EMBED_DIMS = [50, 100, 200]
PHASE2_HIDDEN_DIMS = [64, 128, 256]

#-----------------------------------------------------------------------------
#1. Load BIO Data
#Each address is separated by a blank line

def load_bio(path):
    sentences = []
    tokens = []
    tags = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            #Blank line marks the end of one address sequence
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens = []
                    tags = []
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Unexpected line format in {path}: {line}")

            token, tag = parts
            tokens.append(token)
            tags.append(tag)

    #Add final sequence if the file does not end with a blank line
    if tokens:
        sentences.append((tokens, tags))

    return sentences

#Loading fixed train, development, and test splits
train_data = load_bio(TRAIN_FILE)
dev_data = load_bio(DEV_FILE)
test_data = load_bio(TEST_FILE)

print("Train sequences:", len(train_data))
print("Dev sequences:", len(dev_data))
print("Test sequences:", len(test_data))

#-----------------------------------------------------------------------------
#2. Build vocabularies
word_counter = Counter()
tag_set = set()

#Vocabulary and tag set are built from the training data only
for tokens, tags in train_data:
    word_counter.update(tokens)
    tag_set.update(tags)

#Reserving 0 for padding and 1 for unknown tokens
word2idx = {w: i + 2 for i, w in enumerate(word_counter)}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

#BIO tag mapping
tag2idx = {tag: i for i, tag in enumerate(sorted(tag_set))}
idx2tag = {i: tag for tag, i in tag2idx.items()}

print("Vocabulary size:", len(word2idx))
print("Number of BIO tags:", len(tag2idx))

#Converting tokens to integer IDs
def encode_sentence(tokens):
    return [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]

#Converting BIO tags to integer IDs
def encode_tags(tags):
    return [tag2idx[tag] for tag in tags]

#-----------------------------------------------------------------------------
#3. Dataset and Batching
#PyTorch dataset wrapper for token and BIO-tag sequences

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        x = torch.tensor(encode_sentence(tokens), dtype=torch.long)
        y = torch.tensor(encode_tags(tags), dtype=torch.long)
        return x, y

#Padding variable-length sequences within a batch
#Mask also created so the CRF ignores padding positions
def collate(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    padded_x = []
    padded_y = []
    mask = []

    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)

        padded_x.append(torch.cat([x, torch.zeros(pad_len, dtype=torch.long)]))
        padded_y.append(torch.cat([y, torch.zeros(pad_len, dtype=torch.long)]))
        mask.append(torch.cat([torch.ones(len(x)), torch.zeros(pad_len)]))

    return (
        torch.stack(padded_x).to(DEVICE),
        torch.stack(padded_y).to(DEVICE),
        torch.stack(mask).bool().to(DEVICE),
    )

#Creating DataLoaders for train, dev, test sets
def make_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        NERDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    dev_loader = torch.utils.data.DataLoader(
        NERDataset(dev_data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )

    test_loader = torch.utils.data.DataLoader(
        NERDataset(test_data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )

    return train_loader, dev_loader, test_loader

#-----------------------------------------------------------------------------
#4. Model Definition

#BiLSTM-CRF sequence labelling model for BIO-tag prediction
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim, hidden_dim):
        super().__init__()

        #Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        #Bidrectional LSTM encodes left and right context
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        #Linear layer maps LSTM output to BIO tag scores
        self.fc = nn.Linear(hidden_dim, tagset_size)

        #CRF layer models valid tag transitions
        self.crf = CRF(tagset_size)


    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        emissions = self.fc(lstm_out)
        return emissions

    def loss(self, emissions, tags, mask):
        #Negative log-likelihood loss from the CRF
        ll = self.crf(emissions, tags, mask=mask)
        if ll.ndim > 0:
            return -ll.mean()
        return -ll

    def decode(self, emissions, mask):
        #Viterbi decoding returns the most likely BIO tag sequence
        return self.crf.viterbi_decode(emissions, mask)

#-----------------------------------------------------------------------------
#5. Evaluation

#Evaluate model predictions using entity-level seqeval metrics
def evaluate_entity_level(model, loader):
    model.eval()

    pred_sequences = []
    gold_sequences = []

    with torch.no_grad():
        for x, y, mask in loader:
            emissions = model(x)
            decoded = model.decode(emissions, mask)

            for pred_seq, gold_seq, m in zip(decoded, y, mask):
                length = int(m.sum().item())

                pred_tags = [idx2tag[i] for i in pred_seq[:length]]
                gold_tags = [idx2tag[i] for i in gold_seq[:length].tolist()]

                pred_sequences.append(pred_tags)
                gold_sequences.append(gold_tags)

    precision = precision_score(gold_sequences, pred_sequences)
    recall = recall_score(gold_sequences, pred_sequences)
    f1 = f1_score(gold_sequences, pred_sequences)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_sequences": gold_sequences,
        "pred_sequences": pred_sequences,
    }


#Printing entity-level summary and detailed classification report
def print_detailed_report(model, loader, name):
    scores = evaluate_entity_level(model, loader)

    print("\n==============================")
    print(name, "ENTITY LEVEL RESULTS")
    print("==============================")
    print("Precision:", round(scores["precision"], 4))
    print("Recall:", round(scores["recall"], 4))
    print("F1:", round(scores["f1"], 4))

    print("\nDetailed report:\n")
    print(seqeval_classification_report(scores["gold_sequences"], scores["pred_sequences"]))

#-----------------------------------------------------------------------------
#6. Training Function

#Train one BiLSTM-CRF configuration and return its best development result
def train_one_config(embed_dim, hidden_dim, epochs, learning_rate, batch_size):
    train_loader, dev_loader, _ = make_loaders(batch_size)

    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_dev_f1 = -1.0
    best_dev_scores = None
    best_state = None
    best_epoch = -1

    epoch_times = []
    experiment_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        model.train()
        total_loss = 0.0

        for x, y, mask in train_loader:
            optimizer.zero_grad()

            emissions = model(x)
            loss = model.loss(emissions, y, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #Evaluate after every epoch
        dev_scores = evaluate_entity_level(model, dev_loader)

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"[emb={embed_dim}, hid={hidden_dim}, epochs={epochs}, "
            f"lr={learning_rate}, batch={batch_size}] "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss:.4f} | "
            f"Dev P: {dev_scores['precision']:.4f} | "
            f"Dev R: {dev_scores['recall']:.4f} | "
            f"Dev F1: {dev_scores['f1']:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        #Keep the best epoch based on development F1
        if dev_scores["f1"] > best_dev_f1:
            best_dev_f1 = dev_scores["f1"]
            best_dev_scores = dev_scores
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    total_experiment_time = time.perf_counter() - experiment_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "dev_precision": best_dev_scores["precision"],
        "dev_recall": best_dev_scores["recall"],
        "dev_f1": best_dev_scores["f1"],
        "total_experiment_time_sec": total_experiment_time,
        "avg_epoch_time_sec": avg_epoch_time,
    }


#Loading the best model configuration and evaluating it on the test set
def evaluate_test(best_config, state_dict):
    test_start = time.perf_counter()

    _, _, test_loader = make_loaders(best_config["batch_size"])

    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        embed_dim=best_config["embed_dim"],
        hidden_dim=best_config["hidden_dim"]
    ).to(DEVICE)

    model.load_state_dict(state_dict)

    scores = evaluate_entity_level(model, test_loader)
    test_time = time.perf_counter() - test_start

    return model, scores, test_loader, test_time

#-----------------------------------------------------------------------------
#7. Main Hyperparameter Search Workflow

#Running two-phase hyperparameter search and final tesst evaluation
def main():
    overall_start = time.perf_counter()

    results = []

    best_overall = None
    best_state = None
    best_dev_f1 = -1.0

    #PHASE 1: Tune training settings with fixed model dimensions
    for epochs in PHASE1_EPOCHS:
        for learning_rate in PHASE1_LEARNING_RATES:
            for batch_size in PHASE1_BATCH_SIZES:
                print("\n" + "=" * 80)
                print(
                    f"PHASE 1 | embed_dim={PHASE1_EMBED_DIM}, hidden_dim={PHASE1_HIDDEN_DIM}, "
                    f"epochs={epochs}, lr={learning_rate}, batch_size={batch_size}"
                )

                result = train_one_config(
                    embed_dim=PHASE1_EMBED_DIM,
                    hidden_dim=PHASE1_HIDDEN_DIM,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )

                row = {
                    "phase": 1,
                    "embed_dim": PHASE1_EMBED_DIM,
                    "hidden_dim": PHASE1_HIDDEN_DIM,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "best_epoch": result["best_epoch"],
                    "dev_precision": round(result["dev_precision"], 6),
                    "dev_recall": round(result["dev_recall"], 6),
                    "dev_f1": round(result["dev_f1"], 6),
                    "total_experiment_time_sec": round(result["total_experiment_time_sec"], 2),
                    "avg_epoch_time_sec": round(result["avg_epoch_time_sec"], 2),
                }
                results.append(row)

                #Track best configuration across all experiments
                if result["dev_f1"] > best_dev_f1:
                    best_dev_f1 = result["dev_f1"]
                    best_overall = {
                        "embed_dim": PHASE1_EMBED_DIM,
                        "hidden_dim": PHASE1_HIDDEN_DIM,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "best_epoch": result["best_epoch"],
                        "total_experiment_time_sec": result["total_experiment_time_sec"],
                        "avg_epoch_time_sec": result["avg_epoch_time_sec"],
                    }
                    best_state = result["best_state"]

    print("\nBest configuration after Phase 1:")
    print(best_overall)
    print(f"Best dev F1 after Phase 1: {best_dev_f1:.4f}")

    #Reuse best Phase 1 settings in Phase 2
    fixed_epochs = best_overall["epochs"]
    fixed_learning_rate = best_overall["learning_rate"]
    fixed_batch_size = best_overall["batch_size"]

    #PHASE 2: Tune embedding and hidden dimensions
    for embed_dim in PHASE2_EMBED_DIMS:
        for hidden_dim in PHASE2_HIDDEN_DIMS:
            print("\n" + "=" * 80)
            print(
                f"PHASE 2 | embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
                f"epochs={fixed_epochs}, lr={fixed_learning_rate}, batch_size={fixed_batch_size}"
            )

            result = train_one_config(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                epochs=fixed_epochs,
                learning_rate=fixed_learning_rate,
                batch_size=fixed_batch_size
            )

            row = {
                "phase": 2,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "epochs": fixed_epochs,
                "learning_rate": fixed_learning_rate,
                "batch_size": fixed_batch_size,
                "best_epoch": result["best_epoch"],
                "dev_precision": round(result["dev_precision"], 6),
                "dev_recall": round(result["dev_recall"], 6),
                "dev_f1": round(result["dev_f1"], 6),
                "total_experiment_time_sec": round(result["total_experiment_time_sec"], 2),
                "avg_epoch_time_sec": round(result["avg_epoch_time_sec"], 2),
            }
            results.append(row)

            #Update best overall model if Phase 2 improves dev F1
            if result["dev_f1"] > best_dev_f1:
                best_dev_f1 = result["dev_f1"]
                best_overall = {
                    "embed_dim": embed_dim,
                    "hidden_dim": hidden_dim,
                    "epochs": fixed_epochs,
                    "learning_rate": fixed_learning_rate,
                    "batch_size": fixed_batch_size,
                    "best_epoch": result["best_epoch"],
                    "total_experiment_time_sec": result["total_experiment_time_sec"],
                    "avg_epoch_time_sec": result["avg_epoch_time_sec"],
                }
                best_state = result["best_state"]

    #Save all hyperparameter seach results
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "embed_dim",
                "hidden_dim",
                "epochs",
                "learning_rate",
                "batch_size",
                "best_epoch",
                "dev_precision",
                "dev_recall",
                "dev_f1",
                "total_experiment_time_sec",
                "avg_epoch_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved hyperparameter search results to:")
    print(RESULTS_CSV)

    #Save best model checkpoint and vocabularies
    torch.save(
        {
            "model_state_dict": best_state,
            "best_config": best_overall,
            "best_dev_f1": best_dev_f1,
            "word2idx": word2idx,
            "tag2idx": tag2idx,
        },
        BEST_MODEL_FILE
    )

    print("\n=== Best Overall Configuration ===")
    print(best_overall)
    print(f"Best dev F1: {best_dev_f1:.4f}")
    print(f"Best experiment total time: {best_overall['total_experiment_time_sec']:.2f} seconds")
    print(f"Best experiment average epoch time: {best_overall['avg_epoch_time_sec']:.2f} seconds")

    #Final evaluation on held-out test data
    best_model, test_scores, test_loader, test_time = evaluate_test(best_overall, best_state)

    print("\n=== Final Test Results ===")
    print(f"Precision: {test_scores['precision']:.4f}")
    print(f"Recall:    {test_scores['recall']:.4f}")
    print(f"F1:        {test_scores['f1']:.4f}")
    print(f"Test evaluation time: {test_time:.2f} seconds")

    print_detailed_report(best_model, test_loader, "TEST")

    total_runtime = time.perf_counter() - overall_start
    print(f"\nTotal search + final evaluation time: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()