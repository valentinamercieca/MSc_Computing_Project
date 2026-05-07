#Address Reconstruction Analysis using BERTu Predictions

#Using the BERTu token classificaion model to predict address components on the test set
#Predicted BIO tags are converted into address components, standardised, and reconstructed into
#a canonical addres format.

#Outputs:
# - component coverage summary
# - canonical reconstruction completeness statistics
# - reconstructed address examples
#-----------------------------------------------------------------------------

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
)


#-----------------------------------------------------------------------------
#1. PATHS

#Input and Output paths
BASE_DIR = Path(r"D:\ML_Experiments\Final\Outputs")
MODEL_DIR = BASE_DIR / "hf_hpsearch_MLRS_BERTu/best_model/checkpoint-1050"
TEST_BIO_FILE = BASE_DIR / "test.txt"

#Directory for reconstruction analysis outputs
OUTPUT_DIR = BASE_DIR / "bertu_reconstruction_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#-----------------------------------------------------------------------------
#2. LOAD BIO FILE

#Loading BIO-formatted test data
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
                    records.append({"tokens": tokens, "gold_tags": tags})
                    tokens = []
                    tags = []
                continue

            parts = line.split()
            token = parts[0]
            tag = parts[-1]

            tokens.append(token)
            tags.append(tag)

    #Add final sequence if the file does not end with a blank line
    if tokens:
        records.append({"tokens": tokens, "gold_tags": tags})

    return records


test_records = load_bio_file(TEST_BIO_FILE)
print("Loaded test sequences:", len(test_records))


#-----------------------------------------------------------------------------
#3. LABEL MAPS

#Labels are derived from the test file for decoding aligned gold labels
unique_labels = sorted({tag for row in test_records for tag in row["gold_tags"]})
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print("Labels:", unique_labels)


#-----------------------------------------------------------------------------
#4. HF DATASET

#Converting test records into Hugging Face Dataset format
test_dataset = Dataset.from_list(
    [{"tokens": row["tokens"], "ner_tags": row["gold_tags"]} for row in test_records]
)


#-----------------------------------------------------------------------------
#5. LOAD TOKENIZER AND MODEL

#Loading trained tokeniser and model checkpoint
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

#-----------------------------------------------------------------------------
#6. TOKENIZE AND ALIGN LABELS

#Tokenising word-level BIO data and aligning labels to transformer subwords
#Only the first subword of each token receives a label. Special tokens and non-initial subwords are
#assigned -100 and ignored during prediction output reconstruction
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
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


test_tokenized = test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=test_dataset.column_names,
)


#-----------------------------------------------------------------------------
#7. RUN INFERENCE

#Data collator applies dynamic padding during prediction
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    data_collator=data_collator,
)

test_results = trainer.predict(test_tokenized)


#-----------------------------------------------------------------------------
#8. CONVERT SUBWORD PREDICTIONS TO WORD-LEVEL BIO TAGS

#Converting model logits into word-level BIO tag sequences
#Positions labelled -100 are skipped so output aligns with original tokens
def get_predicted_tag_sequences(pred_output):
    logits = pred_output.predictions
    labels = pred_output.label_ids
    predictions = np.argmax(logits, axis=2)

    pred_sequences = []

    for pred_seq, label_seq in zip(predictions, labels):
        current_preds = []

        for pred_id, label_id in zip(pred_seq, label_seq):

            #Ignore special tokens and non-initial subwords
            if label_id == -100:
                continue
            current_preds.append(model.config.id2label[int(pred_id)])

        pred_sequences.append(current_preds)

    return pred_sequences


predicted_tag_sequences = get_predicted_tag_sequences(test_results)
print("Predicted sequences:", len(predicted_tag_sequences))


#-----------------------------------------------------------------------------
#9. BIO TO ENTITY DICTIONARY

#Converting BIO tags into a dictionary of predicted address components
#Repeated components of the same type are concatenated using a separator
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
                #Treat inconsistent I-tags as a new component
                if current_label is not None:
                    entity_text = " ".join(current_tokens).strip()
                    if current_label in entities:
                        entities[current_label] += " | " + entity_text
                    else:
                        entities[current_label] = entity_text

                current_label = label
                current_tokens = [token]

    #Add final open entity span
    if current_label is not None:
        entity_text = " ".join(current_tokens).strip()
        if current_label in entities:
            entities[current_label] += " | " + entity_text
        else:
            entities[current_label] = entity_text

    return entities


#-----------------------------------------------------------------------------
#10. CLEANING AND STANDARDISATION

#Normalising spacing around punctuation and address separators
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

#Standardising forms for common street prefixes
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

#Standardising forms for common sub-unit abbreviations
SUBUNIT_MAP = {
    "FL": "Flat",
    "FLT": "Flat",
    "FLAT": "Flat",
    "APT": "Apartment",
    "APTS": "Apartments",
    "APP": "Apartment",
    "BLK": "Block",
    "BLOKK": "Block",
    "ENT": "Entrance",
    "MAIS": "Maisonette",
    "PH": "Penthouse",
    "P/H": "Penthouse",
    "PENTHOUSE": "Penthouse",
    "RES": "Residence",
    "CRT": "Court",
    "HSE": "House",
}


#Standardising predicted street prefix values
def standardise_street_prefix(text):
    text = clean_spacing(text).upper()
    return PREFIX_MAP.get(text, normalise_case(text))


#Standardising predicted sub-unit values and abbreviations
def standardise_sub_unit(text):
    text = clean_spacing(text)
    parts = text.split()
    new_parts = []

    for part in parts:
        key = part.upper()
        if key in SUBUNIT_MAP:
            new_parts.append(SUBUNIT_MAP[key])
        else:
            new_parts.append(normalise_case(part))

    return " ".join(new_parts).strip()

#Applying component-specific standardisation rules
def standardise_component(label, text):
    if text is None:
        return None

    text = clean_spacing(text)

    if label == "STREET_PREFIX":
        return standardise_street_prefix(text)

    if label == "SUB_UNIT":
        return standardise_sub_unit(text)

    if label == "PROPERTY_NUMBER":
        return text.strip()

    return normalise_case(text)


#Standardising all extracted address components
def standardise_address_dict(entity_dict):
    standardised = {}
    for label, value in entity_dict.items():
        standardised[label] = standardise_component(label, value)
    return standardised


#-----------------------------------------------------------------------------
#11. CANONICAL ORDER AND RECONSTRUCTION

CANONICAL_ORDER = [
    "PROPERTY_NUMBER",
    "PROPERTY_NAME",
    "SUB_UNIT",
    "STREET_PREFIX",
    "STREET_NAME",
    "LOCALITY",
    "ISLAND",
]

#Returning components in the predefined canonical order
def order_components(entity_dict):
    ordered = {}
    for key in CANONICAL_ORDER:
        if key in entity_dict:
            ordered[key] = entity_dict[key]
    return ordered

#Building a readable canonical address string in a fixed order
def reconstruct_canonical_address(entity_dict):
    parts = []

    #Property-level components
    line1 = []
    for key in ["PROPERTY_NUMBER", "PROPERTY_NAME", "SUB_UNIT"]:
        if key in entity_dict and entity_dict[key]:
            line1.append(entity_dict[key])

    #Street-level components
    line2 = []
    for key in ["STREET_PREFIX", "STREET_NAME"]:
        if key in entity_dict and entity_dict[key]:
            line2.append(entity_dict[key])

    #Locality-level components
    line3 = []
    for key in ["LOCALITY", "ISLAND"]:
        if key in entity_dict and entity_dict[key]:
            line3.append(entity_dict[key])

    if line1:
        parts.append(", ".join(line1))
    if line2:
        parts.append(" ".join(line2))
    if line3:
        parts.append(", ".join(line3))

    return " | ".join(parts)

#Reconstructing an approximate readable address from token sequence
def reconstruct_raw_address_from_tokens(tokens):
    text = " ".join(tokens)
    return clean_spacing(text)


#-----------------------------------------------------------------------------
#12. BUILD RECONSTRUCTION OUTPUTS

analysis_rows = []

for record, pred_tags in zip(test_records, predicted_tag_sequences):
    tokens = record["tokens"]

    #Keep token and prediction lengths aligned
    used_len = min(len(tokens), len(pred_tags))

    raw_extracted = bio_to_entities(tokens[:used_len], pred_tags[:used_len])
    raw_extracted = {k: clean_spacing(v) for k, v in raw_extracted.items()}
    raw_extracted = order_components(raw_extracted)

    standardised = standardise_address_dict(raw_extracted)
    standardised = order_components(standardised)

    canonical_address = reconstruct_canonical_address(standardised)

    analysis_rows.append({
        "raw_address": reconstruct_raw_address_from_tokens(tokens[:used_len]),
        "predicted_tags": pred_tags[:used_len],
        "raw_extracted": raw_extracted,
        "standardised_extracted": standardised,
        "canonical_address": canonical_address,
    })


#-----------------------------------------------------------------------------
#13. COMPONENT COVERAGE ANALYSIS

coverage_counts = Counter()

for row in analysis_rows:
    for component in CANONICAL_ORDER:
        if component in row["standardised_extracted"] and row["standardised_extracted"][component]:
            coverage_counts[component] += 1

print("\n==================================================")
print("COMPONENT COVERAGE ANALYSIS")
print("==================================================")
print("Addresses processed:", len(analysis_rows))

coverage_summary = {}
for component in CANONICAL_ORDER:
    count = coverage_counts.get(component, 0)
    pct = count / len(analysis_rows)
    coverage_summary[component] = {
        "count": count,
        "percentage": pct,
    }
    print(f"{component:16} count={count:4d} | pct={pct:.4f}")


#-----------------------------------------------------------------------------
#14. CANONICAL RECONSTRUCTION COMPLETENESS

#Core address requires street prefix, street name, locality, and island
core_components = ["STREET_PREFIX", "STREET_NAME", "LOCALITY", "ISLAND"]

complete_core = 0
complete_extended = 0

for row in analysis_rows:
    d = row["standardised_extracted"]

    if all(comp in d and d[comp] for comp in core_components):
        complete_core += 1

    #Extended address also requires at least one property-level component
    has_property_level = any(
        comp in d and d[comp]
        for comp in ["PROPERTY_NUMBER", "PROPERTY_NAME", "SUB_UNIT"]
    )
    if all(comp in d and d[comp] for comp in core_components) and has_property_level:
        complete_extended += 1

print("\n==================================================")
print("CANONICAL RECONSTRUCTION COMPLETENESS")
print("==================================================")
print("Core complete addresses:     ", complete_core, f"({complete_core / len(analysis_rows):.4f})")
print("Extended complete addresses: ", complete_extended, f"({complete_extended / len(analysis_rows):.4f})")


#-----------------------------------------------------------------------------
#15. SHOW EXAMPLES

print("\n==================================================")
print("CANONICAL ADDRESS EXAMPLES")
print("==================================================")

for i, row in enumerate(analysis_rows[:10], start=1):
    print(f"\n--- Example {i} ---")
    print("Original test address: ", row["raw_address"])
    print("Extracted components:  ", row["standardised_extracted"])
    print("Canonical address:     ", row["canonical_address"])


#-----------------------------------------------------------------------------
#16. SAVE OUTPUTS

summary = {
    "addresses_processed": len(analysis_rows),
    "component_coverage": coverage_summary,
    "core_complete_count": complete_core,
    "core_complete_proportion": complete_core / len(analysis_rows),
    "extended_complete_count": complete_extended,
    "extended_complete_proportion": complete_extended / len(analysis_rows),
}

#Saving summary statistics
with open(OUTPUT_DIR / "reconstruction_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

#Saving address-level reconstruction outputs
with open(OUTPUT_DIR / "reconstructed_addresses.json", "w", encoding="utf-8") as f:
    json.dump(analysis_rows, f, indent=2, ensure_ascii=False)

print(f"\nSaved outputs to: {OUTPUT_DIR}")