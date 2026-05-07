#Model-assisted pre-annotation for Label Studio (local)

#This script trains a preliminary spaCy NER model using the 600 manually annotated address tasks, 
#then generates pre-annotations for the remaining unannotated tasks.
#The output predictions are imported back into Label Studio for manual review.


from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import spacy
from spacy.training import Example
#-----------------------------------------------------------------------------
#1. PATHS AND BASIC SETTINGS

#Full Label Studio export containing both annotated and unannotated tasks
EXPORT_JSON_PATH = Path(
    r"C:\Users\valen\Desktop\MSc Dissertation\Code\Annotation\Annotated 600 out of 3000.json"
)

#Directory where the trained spaCy model will be saved
MODEL_OUT_DIR = Path(r"C:\Users\valen\Desktop\MSc Dissertation\Code\Models\spacy_ner_600")

#Output file containing predictions for import into Label Studio
PREDICTIONS_JSON_OUT = Path(
    r"C:\Users\valen\Desktop\MSc Dissertation\Code\Annotation\predictions_for_remaining_2400.json"
)

#Label Studio config names (must match XML)
LS_FROM_NAME = "address_labels"
LS_TO_NAME = "addr"

#Training settings used for reproducibility
RANDOM_SEED = 42
DEV_FRAC = 0.2
N_ITER = 20


#-----------------------------------------------------------------------------
#2. LOAD AND PARSE LABEL STUDIO EXPORT

def load_labelstudio_export(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("Expected a list of tasks in the Label Studio export JSON.")
    return tasks


def get_text_from_task(task: Dict[str, Any]) -> str:

    #Extract the address text from Label Studio task
    #Fallback is included in case export uses a different text field name
    data = task.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Task has no valid 'data' dict.")

    if "raw_address_string" in data and isinstance(data["raw_address_string"], str):
        return data["raw_address_string"]

    #Fallback: first string field in data
    for v in data.values():
        if isinstance(v, str):
            return v

    raise ValueError("Could not find a text field in task['data'].")


def extract_entities_from_task(task: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    
    #Extract labelled entity spans from the latest annotation
    #Label Studio stores NER spans as character offsets whichh are preserved here for spaCy training
    annotations = task.get("annotations", [])
    if not annotations:
        return []

    #Use the latest annotation (usually last) for each task
    ann = annotations[-1]
    result = ann.get("result", [])
    ents: List[Tuple[int, int, str]] = []

    for r in result:
        if not isinstance(r, dict):
            continue

        #Keep only NER label results from the expected Label Studio fields
        if r.get("type") != "labels":
            continue
        if r.get("from_name") != LS_FROM_NAME:
            continue
        if r.get("to_name") != LS_TO_NAME:
            continue

        val = r.get("value", {})
        if not isinstance(val, dict):
            continue

        start = val.get("start")
        end = val.get("end")
        labels = val.get("labels")

        if isinstance(start, int) and isinstance(end, int) and isinstance(labels, list) and labels:
            label = labels[0]
            if isinstance(label, str):
                ents.append((start, end, label))

    #Sorting to ensure consistent entity ordering
    ents.sort(key=lambda x: (x[0], x[1]))
    return ents


def split_annotated_unannotated(tasks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    #Separate manually annotated tasks from tasks without submitted annotations
    annotated: List[Dict[str, Any]] = []
    unannotated: List[Dict[str, Any]] = []

    for t in tasks:
        anns = t.get("annotations", [])
        is_annot = any(isinstance(a, dict) and a.get("result") for a in anns)
        if is_annot:
            annotated.append(t)
        else:
            unannotated.append(t)

    return annotated, unannotated


#-----------------------------------------------------------------------------
#3. TRAIN SPACY NER MODEL


def train_spacy_ner(annotated_tasks: List[Dict[str, Any]], model_out_dir: Path) -> spacy.language.Language:
    #Train a preliminary spaCy NER model from manually annotated address spans.
    random.seed(RANDOM_SEED)

    #Store training pairs as text plus character-offset entities
    train_examples: List[Tuple[str, List[Tuple[int, int, str]]]] = []
    label_set = set()

    for t in annotated_tasks:
        text = get_text_from_task(t)
        ents = extract_entities_from_task(t)
        if not text or not ents:
            continue
        for _, _, lab in ents:
            label_set.add(lab)
        train_examples.append((text, ents))

    if not train_examples:
        raise ValueError("No training examples found. Check your export JSON content.")

    #Train/dev split
    random.shuffle(train_examples)
    n_dev = max(1, int(len(train_examples) * DEV_FRAC))
    dev_data = train_examples[:n_dev]
    train_data = train_examples[n_dev:]

    print("Annotated tasks provided:", len(annotated_tasks))
    print("Training examples used:", len(train_data))
    print("Dev examples used:", len(dev_data))
    print("Labels:", sorted(label_set))

    #Blank English pipeline is sufficient because training uses character spans
    nlp = spacy.blank("en")

    #Add NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    #Register all entity labels found in the annotation set
    for lab in sorted(label_set):
        ner.add_label(lab)

    #Convert text and entity spans into spaCy Example objects
    def make_examples(pairs: List[Tuple[str, List[Tuple[int, int, str]]]]) -> List[Example]:
        exs: List[Example] = []
        for text, ents in pairs:
            doc = nlp.make_doc(text)
            exs.append(Example.from_dict(doc, {"entities": ents}))
        return exs

    train_exs = make_examples(train_data)
    dev_exs = make_examples(dev_data)

    #Train
    optimizer = nlp.begin_training()
    for i in range(1, N_ITER + 1):
        random.shuffle(train_exs)
        losses: Dict[str, float] = {}
        for ex in train_exs:
            nlp.update([ex], sgd=optimizer, losses=losses)

        #Print occasional progress without overcrowding the console
        if i % 5 == 0 or i == 1:
            print(f"Iter {i}/{N_ITER} Losses:", losses)

    #Development-set evaluation
    scorer = nlp.evaluate(dev_exs)
    print("\nDEV (quick):")
    print("ents_p:", scorer.get("ents_p"))
    print("ents_r:", scorer.get("ents_r"))
    print("ents_f:", scorer.get("ents_f"))

    #Save trained model for reproducibility and reuse
    model_out_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_out_dir)
    print("\nSaved model to:", str(model_out_dir))

    return nlp


#-----------------------------------------------------------------------------
#4. GENERATE LABEL STUDIO PREDICTIONS

#Build one Label Studio prediction result for a single entity span
def make_ls_prediction_result(text: str, start: int, end: int, label: str) -> Dict[str, Any]:

    return {
        "id": str(uuid.uuid4())[:8],
        "from_name": LS_FROM_NAME,
        "to_name": LS_TO_NAME,
        "type": "labels",
        "value": {
            "start": start,
            "end": end,
            "text": text[start:end],
            "labels": [label],
        },
    }

#Generate a Label Studio predictions JSON file for unannotated tasks
def build_predictions_import(
    nlp: spacy.language.Language,
    unannotated_tasks: List[Dict[str, Any]],
    out_path: Path,
    model_version: str = "spacy_ner_600_v1",
) -> None:
    
    #Task IDs are preserved so Label Studio can attach predictions to the existing tasks
    payload: List[Dict[str, Any]] = []

    for t in unannotated_tasks:
        task_id = t.get("id")
        if task_id is None:
            continue

        text = get_text_from_task(t)
        doc = nlp(text)

        results = []

        #Convert spaCy entity predictions into Label Studio span format
        for ent in doc.ents:
            results.append(make_ls_prediction_result(text, ent.start_char, ent.end_char, ent.label_))

        item = {
            "id": task_id,
            "predictions": [
                {
                    "model_version": model_version,
                    "result": results,
                }
            ],
        }
        payload.append(item)

    #Save predictions in Label Studio import format
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print("\nPredictions file written to:", str(out_path))
    print("Tasks with predictions:", len(payload))


#-----------------------------------------------------------------------------
#5. MAIN WORKFLOW

def main() -> None:
    #Load Label Studio export
    tasks = load_labelstudio_export(EXPORT_JSON_PATH)
    print("Total tasks loaded:", len(tasks))

    #Split annotated and unannotated tasks
    annotated, unannotated = split_annotated_unannotated(tasks)
    print("Annotated tasks:", len(annotated))
    print("Unannotated tasks:", len(unannotated))

    #Train spaCy NER model
    nlp = train_spacy_ner(annotated, MODEL_OUT_DIR)

    #Build predictions JSON for unannotated tasks
    build_predictions_import(nlp, unannotated, PREDICTIONS_JSON_OUT)

if __name__ == "__main__":
    main()
