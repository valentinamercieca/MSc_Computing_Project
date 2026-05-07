#APPLYING TRAINED MODEL TO FULL ADDRESS DATASET

#Applying the trained BERTu model to the full dataset of raw address strings
#Duplicate address strings are parsed only once, then merged back to the full dataset
#Predicted components are standardised and reconstructed into a canonical address format

#Outputs:
# - Full parsed dataset
# - Sample parsed outputs
# - Descriptive summary JSON
#-----------------------------------------------------------------------------

import json
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
)

#-----------------------------------------------------------------------------
#1. USER SETTINGS

#Input dataset containing raw address strings
INPUT_FILE = Path(r"D:\ML_Experiments\Final\Outputs\full_data_inc_raw_addresses.csv")
ADDRESS_COLUMN = "raw_address_string"

#Trained BERTu model checkpoint
MODEL_DIR = Path(r"D:\ML_Experiments\Final\Outputs\hf_hpsearch_MLRS_BERTu/best_model/checkpoint-1050")

#Output directory for parsed full-dataset results
OUTPUT_DIR = Path(r"D:\ML_Experiments\Final\Outputs\full_dataset_application")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Batch size used during inference
BATCH_SIZE = 16

#-----------------------------------------------------------------------------
#2. TIMING

script_start = time.perf_counter()
timings = {}

#Storing elapsed time for one processing step
def mark_time(step_name: str, start_time: float) -> None:
    timings[step_name] = round(time.perf_counter() - start_time, 2)

#-----------------------------------------------------------------------------
#3. LOAD DATASET

t0 = time.perf_counter()

df = pd.read_csv(INPUT_FILE)

if ADDRESS_COLUMN not in df.columns:
    raise ValueError(f"Column '{ADDRESS_COLUMN}' not found. Available columns: {list(df.columns)}")

original_row_count = len(df)

#Keeping rows with usable raw address strings
df = df[df[ADDRESS_COLUMN].notna()].copy()
df[ADDRESS_COLUMN] = df[ADDRESS_COLUMN].astype(str).str.strip()
df = df[df[ADDRESS_COLUMN] != ""].copy()

rows_with_non_empty_addresses = len(df)

mark_time("load_and_filter_data_sec", t0)

print("Original rows:", original_row_count)
print("Rows with non-empty addresses:", rows_with_non_empty_addresses)

#-----------------------------------------------------------------------------
#4. DEDUPLICATION

t0 = time.perf_counter()

#Parsing each unique address once, then merging results back later
unique_df = df[[ADDRESS_COLUMN]].drop_duplicates().reset_index(drop=True)
unique_address_count = len(unique_df)

mark_time("deduplication_sec", t0)

print("Unique addresses to parse:", unique_address_count)

#-----------------------------------------------------------------------------
#5. TOKENISER

#Token pattern aligned with the BIO conversion and modelling pipeline
TOKEN_PATTERN = re.compile(
    r"[A-Za-zÀ-ÿ]+-|[A-Za-zÀ-ÿ]+\.(?=\s|,|$)|[A-Za-zÀ-ÿ0-9]+(?:['’`][A-Za-zÀ-ÿ0-9]+)*`?|[0-9]+(?:[-/][0-9A-Za-z]+)*|[&/]+|[^\w\s]",
    re.UNICODE
)

#Tokenising address text while preserving address-specific punctuation
def tokenize_address(text: str):
    return TOKEN_PATTERN.findall(text)

t0 = time.perf_counter()

unique_df["tokens"] = unique_df[ADDRESS_COLUMN].apply(tokenize_address)
unique_df = unique_df[unique_df["tokens"].map(len) > 0].reset_index(drop=True)

mark_time("tokenisation_sec", t0)

print("Tokenised unique addresses:", len(unique_df))

#-----------------------------------------------------------------------------
#6. LOAD MODEL AND TOKENIZER

t0 = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)

id2label = model.config.id2label

mark_time("load_model_sec", t0)

#-----------------------------------------------------------------------------
#7. CONVERT TO HF DATASET

hf_dataset = Dataset.from_pandas(unique_df[[ADDRESS_COLUMN, "tokens"]], preserve_index=False)

#Tokenising pre-split address tokens for transformer inference
def tokenize_for_inference(examples):
    return tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )

t0 = time.perf_counter()

tokenized_dataset = hf_dataset.map(
    tokenize_for_inference,
    batched=True,
    remove_columns=hf_dataset.column_names,
)

mark_time("hf_tokenisation_sec", t0)

#-----------------------------------------------------------------------------
#8. RUN INFERENCE

t0 = time.perf_counter()

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    data_collator=data_collator,
)

pred_output = trainer.predict(tokenized_dataset)

mark_time("model_inference_sec", t0)

#-----------------------------------------------------------------------------
#9. CONVERT SUBWORD PREDICTIONS BACK TO WORD TAGS

#Converting transformer subword predictions back to word-level BIO tags
def get_predicted_tag_sequences(pred_output, original_tokens_list):
    logits = pred_output.predictions
    predictions = np.argmax(logits, axis=2)

    pred_sequences = []

    for batch_idx, pred_seq in enumerate(predictions):
        word_ids = tokenizer(
            [original_tokens_list[batch_idx]],
            is_split_into_words=True,
            truncation=True,
            max_length=128,
        ).word_ids(batch_index=0)

        current_preds = []
        previous_word_id = None

        for pred_id, word_id in zip(pred_seq, word_ids):

            #Keep only the first subword prediction for each original token
            if word_id is None:
                continue
            if word_id != previous_word_id:
                current_preds.append(id2label[int(pred_id)])
            previous_word_id = word_id

        pred_sequences.append(current_preds)

    return pred_sequences

t0 = time.perf_counter()

predicted_tag_sequences = get_predicted_tag_sequences(
    pred_output,
    unique_df["tokens"].tolist()
)

mark_time("subword_to_word_alignment_sec", t0)

print("Predicted sequences:", len(predicted_tag_sequences))

#-----------------------------------------------------------------------------
#10. BIO TO ENTITY DICTIONARY

#Converting predicted BIO tags into address component spans
def bio_to_entities(tokens, tags):
    entities = {}
    current_label = None
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_label is not None:
                entity_text = " ".join(current_tokens).strip()
                if current_label in entities:
                    entities[current_label] += " | " + entity_text
                else:
                    entities[current_label] = entity_text
                current_label = None
                current_tokens = []
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B":
            if current_label is not None:
                entity_text = " ".join(current_tokens).strip()
                if current_label in entities:
                    entities[current_label] += " | " + entity_text
                else:
                    entities[current_label] = entity_text
            current_label = label
            current_tokens = [token]

        elif prefix == "I":
            if current_label == label:
                current_tokens.append(token)
            else:
                #Treat inconsistent I-tags as a new component span
                if current_label is not None:
                    entity_text = " ".join(current_tokens).strip()
                    if current_label in entities:
                        entities[current_label] += " | " + entity_text
                    else:
                        entities[current_label] = entity_text
                current_label = label
                current_tokens = [token]

    #Add final open component span
    if current_label is not None:
        entity_text = " ".join(current_tokens).strip()
        if current_label in entities:
            entities[current_label] += " | " + entity_text
        else:
            entities[current_label] = entity_text

    return entities

#-----------------------------------------------------------------------------
#11. STANDARDISATION

#Normalising spacing around punctuation and separators
def clean_spacing(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = re.sub(r"\s+-\s+", "-", text)
    return text.strip()

#Applying title casing while preserving common Maltese particles
def normalise_case(text):
    text = clean_spacing(text)
    text = text.lower().title()

    replacements = [
        (r"\bIl-", "Il-"),
        (r"\bIr-", "Ir-"),
        (r"\bId-", "Id-"),
        (r"\bIx-", "Ix-"),
        (r"\bIn-", "In-"),
        (r"\bIs-", "Is-"),
        (r"\bIt-", "It-"),
        (r"\bIz-", "Iz-"),
        (r"\bTal-", "Tal-"),
        (r"\bTas-", "Tas-"),
        (r"\bTaz-", "Taz-"),
        (r"\bTad-", "Tad-"),
        (r"\bTax-", "Tax-"),
        (r"\bTa'", "Ta'"),
        (r"\bL'", "L'"),
        (r"\bD'", "D'"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    return text.strip()

PREFIX_MAP = {
    "TRIQ": "Triq",
    "SQAQ": "Sqaq",
    "PJAZZA": "Pjazza",
    "PJAZZETTA": "Pjazza",
    "MISRAH": "Misrah",
    "VJAL": "Vjal",
    "TREJQET": "Trejqet",
    "DAWRET": "Dawret",
    "WESGHA": "Wesgha",
}

SUBUNIT_MAP = {
    "FL": "Flat",
    "FLT": "Flat",
    "FLAT": "Flat",
    "APT": "Apartment",
    "BLK": "Block",
    "BLOKK": "Block",
    "ENT": "Entrance",
    "MAIS": "Maisonette",
    "PH": "Penthouse",
    "PENTHOUSE": "Penthouse",
}

#Applying component-specific standardisation rules
def standardise_component(label, text):
    text = clean_spacing(text)

    if label == "STREET_PREFIX":
        return PREFIX_MAP.get(text.upper(), normalise_case(text))

    if label == "SUB_UNIT":
        parts = text.split()
        new_parts = [SUBUNIT_MAP.get(part.upper(), normalise_case(part)) for part in parts]
        return " ".join(new_parts).strip()

    if label == "PROPERTY_NUMBER":
        return text.strip()

    return normalise_case(text)

CANONICAL_ORDER = [
    "PROPERTY_NUMBER",
    "PROPERTY_NAME",
    "SUB_UNIT",
    "STREET_PREFIX",
    "STREET_NAME",
    "LOCALITY",
    "ISLAND",
]

#Returning extracted components in canonical order
def order_components(entity_dict):
    ordered = {}
    for key in CANONICAL_ORDER:
        if key in entity_dict:
            ordered[key] = entity_dict[key]
    return ordered

#Reconstructing a readable canonical address string
def reconstruct_canonical_address(entity_dict):
    line1 = []
    for key in ["PROPERTY_NUMBER", "PROPERTY_NAME", "SUB_UNIT"]:
        if key in entity_dict and entity_dict[key]:
            line1.append(entity_dict[key])

    line2 = []
    for key in ["STREET_PREFIX", "STREET_NAME"]:
        if key in entity_dict and entity_dict[key]:
            line2.append(entity_dict[key])

    line3 = []
    for key in ["LOCALITY", "ISLAND"]:
        if key in entity_dict and entity_dict[key]:
            line3.append(entity_dict[key])

    parts = []
    if line1:
        parts.append(", ".join(line1))
    if line2:
        parts.append(" ".join(line2))
    if line3:
        parts.append(", ".join(line3))

    return " | ".join(parts)

#-----------------------------------------------------------------------------
#12. BUILD STRUCTURED OUTPUT

t0 = time.perf_counter()

rows = []

for raw_address, tokens, pred_tags in zip(
    unique_df[ADDRESS_COLUMN].tolist(),
    unique_df["tokens"].tolist(),
    predicted_tag_sequences
):
    #Keep token and prediction lengths aligned
    used_len = min(len(tokens), len(pred_tags))
    tokens = tokens[:used_len]
    pred_tags = pred_tags[:used_len]

    extracted = bio_to_entities(tokens, pred_tags)
    extracted = {k: standardise_component(k, v) for k, v in extracted.items()}
    extracted = order_components(extracted)

    row = {
        ADDRESS_COLUMN: raw_address,
        "canonical_address": reconstruct_canonical_address(extracted),
    }

    for component in CANONICAL_ORDER:
        row[component] = extracted.get(component, None)

    rows.append(row)

parsed_unique_df = pd.DataFrame(rows)

mark_time("reconstruction_and_standardisation_sec", t0)

print("Parsed unique addresses:", len(parsed_unique_df))

#-----------------------------------------------------------------------------
#13. MERGE BACK TO FULL DATASET

t0 = time.perf_counter()

#Merging parsed unique addresses back onto all original rows
final_df = df.merge(parsed_unique_df, on=ADDRESS_COLUMN, how="left")

mark_time("merge_back_sec", t0)

print("Final merged rows:", len(final_df))

#-----------------------------------------------------------------------------
#14. DESCRIPTIVE SUMMARIES

t0 = time.perf_counter()

summary = {
    "original_rows": int(original_row_count),
    "rows_with_non_empty_addresses": int(rows_with_non_empty_addresses),
    "unique_addresses_processed": int(len(parsed_unique_df)),
    "final_rows_output": int(len(final_df)),
    "deduplication_ratio_unique_over_non_empty": round(len(parsed_unique_df) / rows_with_non_empty_addresses, 4),
    "timings_sec": timings,
    "component_non_null_counts": {},
    "component_non_null_percentages": {},
}

#Counting non-null predicted components in the merged dataset
for component in CANONICAL_ORDER:
    count = int(final_df[component].notna().sum())
    pct = round(count / len(final_df), 4)
    summary["component_non_null_counts"][component] = count
    summary["component_non_null_percentages"][component] = pct

#Core completeness: street + locality + island
core_mask = (
    final_df["STREET_NAME"].notna() &
    final_df["LOCALITY"].notna() &
    final_df["ISLAND"].notna()
)

#Extended completeness: core + at least one property-level component
extended_mask = (
    core_mask &
    (
        final_df["PROPERTY_NUMBER"].notna() |
        final_df["PROPERTY_NAME"].notna() |
        final_df["SUB_UNIT"].notna()
    )
)

summary["core_complete_count"] = int(core_mask.sum())
summary["core_complete_percentage"] = round(core_mask.mean(), 4)

summary["extended_complete_count"] = int(extended_mask.sum())
summary["extended_complete_percentage"] = round(extended_mask.mean(), 4)

#Address length and complexity summaries for unique parsed addresses
parsed_unique_df["token_count"] = unique_df["tokens"].map(len).values
parsed_unique_df["raw_char_count"] = parsed_unique_df[ADDRESS_COLUMN].astype(str).str.len()
parsed_unique_df["canonical_char_count"] = parsed_unique_df["canonical_address"].fillna("").astype(str).str.len()

summary["unique_address_descriptives"] = {
    "avg_token_count": round(parsed_unique_df["token_count"].mean(), 2),
    "median_token_count": float(parsed_unique_df["token_count"].median()),
    "avg_raw_char_count": round(parsed_unique_df["raw_char_count"].mean(), 2),
    "median_raw_char_count": float(parsed_unique_df["raw_char_count"].median()),
    "avg_canonical_char_count": round(parsed_unique_df["canonical_char_count"].mean(), 2),
    "median_canonical_char_count": float(parsed_unique_df["canonical_char_count"].median()),
}

#Most frequent predicted localities
if "LOCALITY" in final_df.columns:
    top_localities = (
        final_df["LOCALITY"]
        .dropna()
        .value_counts()
        .head(10)
        .to_dict()
    )
    summary["top_10_localities"] = top_localities

#Most frequent reconstructed canonical addresses
top_canonical = (
    final_df["canonical_address"]
    .dropna()
    .value_counts()
    .head(10)
    .to_dict()
)
summary["top_10_canonical_addresses"] = top_canonical

mark_time("descriptive_summary_sec", t0)

#-----------------------------------------------------------------------------
#15. SAVE OUTPUTS

t0 = time.perf_counter()

#Full parsed dataset
full_output_path = OUTPUT_DIR / "full_dataset_parsed.csv"
final_df.to_csv(full_output_path, index=False, encoding="utf-8")

#Head sample (first 10 rows)
sample_output_path = OUTPUT_DIR / "full_dataset_sample.csv"
final_df[[ADDRESS_COLUMN, "canonical_address"] + CANONICAL_ORDER].head(10).to_csv(
    sample_output_path, index=False, encoding="utf-8"
)

#Random sample (more representative examples)
random_sample_output_path = OUTPUT_DIR / "full_dataset_random_sample.csv"
final_df[[ADDRESS_COLUMN, "canonical_address"] + CANONICAL_ORDER].sample(
    n=10,
    random_state=42
).to_csv(random_sample_output_path, index=False, encoding="utf-8")

#Summary statistics (JSON)
summary_path = OUTPUT_DIR / "full_dataset_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

mark_time("save_outputs_sec", t0)

#Final total runtime
summary["timings_sec"]["total_script_runtime_sec"] = round(time.perf_counter() - script_start, 2)

#Re-save summary with total runtime included
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved outputs:")
print(full_output_path)
print(sample_output_path)
print(random_sample_output_path)
print(summary_path)

print("\nSummary:")
print(json.dumps(summary, indent=2))

