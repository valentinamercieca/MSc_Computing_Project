#Converting Label Studio Annotations to BIO Format
#Final Label Studio NER annotations into token-level BIO format
#For model training and evaluation
#-----------------------------------------------------------------------------

#Importing libraries
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

#Dataset Labels
VALID_LABELS = {
    "PROPERTY_NUMBER",
    "PROPERTY_NAME",
    "SUB_UNIT",
    "STREET_PREFIX",
    "STREET_NAME",
    "LOCALITY",
    "ISLAND"
}

#Token pattern designed for Maltese address text
#It preserves particles, apostrophes, initials, numbers, slashes, and punctuation
TOKEN_PATTERN = re.compile(
    r"[A-Za-zÀ-ÿ]+-|[A-Za-zÀ-ÿ]+\.(?=\s|,|$)|[A-Za-zÀ-ÿ0-9]+(?:['’`][A-Za-zÀ-ÿ0-9]+)*`?|[0-9]+(?:[-/][0-9A-Za-z]+)*|[&/]+|[^\w\s]",
    re.UNICODE
)

#Loading the final Label Studio annotation export
def load_labelstudio_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#Retrieving the first annotation object for a task
def get_first_annotation(task: Dict[str, Any]) -> Dict[str, Any] | None:
    annotations = task.get("annotations", [])
    if not annotations:
        return None
    return annotations[0]

#Extracting labelled character spans from Label Studio results
def extract_entities(task: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    annotation = get_first_annotation(task)
    if annotation is None:
        return []

    entities = []
    for item in annotation.get("result", []):
        if item.get("type") != "labels":
            continue

        value = item.get("value", {})
        start = value.get("start")
        end = value.get("end")
        labels = value.get("labels", [])

        if start is None or end is None or not labels:
            continue

        label = labels[0]

        #Stop execution if the export contains an unexpected entity label
        if label not in VALID_LABELS:
            raise ValueError(f"Unexpected label: {label}")

        entities.append((start, end, label))

    #Sort spans to support overlap validation and consistent conversion
    entities.sort(key=lambda x: (x[0], x[1]))
    return entities

#Check that spans are valid and non-overlapping before conversion
def validate_entities(text: str, entities: List[Tuple[int, int, str]]) -> None:
    prev_end = -1
    for start, end, label in entities:
        if start < 0 or end > len(text) or start >= end:
            raise ValueError(f"Invalid span ({start}, {end}, {label}) in text: {text}")
        if start < prev_end:
            raise ValueError(f"Overlapping spans detected in text: {text}")
        prev_end = end

#Tokenising text while preserving character start and end offsets
def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        start, end = match.span()
        tokens.append((token, start, end))
    return tokens

#Assigning a BIO tag to a token based on its position inside an entity span
def token_tag(token_start: int, token_end: int, entities: List[Tuple[int, int, str]]) -> str:
    for ent_start, ent_end, label in entities:
        
        #Token must be fully inside entity span
        if token_start >= ent_start and token_end <= ent_end:
            return f"B-{label}" if token_start == ent_start else f"I-{label}"
    return "O"

#Converting one Label Studio task into token and BIO-tag pairs
def convert_task_to_bio(task: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
    text = task["data"]["raw_address_string"]
    entities = extract_entities(task)
    validate_entities(text, entities)

    token_spans = tokenize_with_spans(text)
    bio_pairs = []

    for token, start, end in token_spans:
        tag = token_tag(start, end, entities)
        bio_pairs.append((token, tag))

    return text, bio_pairs

#Writing all converted tasks to a BIO-formatted text file
def write_bio_file(tasks: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            text, bio_pairs = convert_task_to_bio(task)
            for token, tag in bio_pairs:
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

#Print sample conversions to visually inspect token and tag alignment
def preview_examples(tasks: List[Dict[str, Any]], n: int = 5) -> None:
    for i, task in enumerate(tasks[:n], start=1):
        text, bio_pairs = convert_task_to_bio(task)
        print(f"\nExample {i}")
        print(text)
        for token, tag in bio_pairs:
            print(f"{token:20} {tag}")

#Main workflow
if __name__ == "__main__":
    input_file = r"C:\Users\valen\Desktop\MSc Dissertation\Code\Annotation\Annotations Final\Annotation_3000.json"
    output_file = r"C:\Users\valen\Desktop\MSc Dissertation\Code\FINAL\Outputs\addresses_bio.txt"

    tasks = load_labelstudio_json(input_file)
    preview_examples(tasks, n=5)
    write_bio_file(tasks, output_file)