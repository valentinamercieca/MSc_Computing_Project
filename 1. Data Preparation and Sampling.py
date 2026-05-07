#DATA PREPARATION AND SAMPLING
#Data Exploration and Preliminary Analysis

#What does the address data look like, and how should that inform annotation and modelling design decisions?

#Purpose:
# - To assess missingness and sparsity in address-related fields
# - To identify fields that should be excluded from modelling
# - To construct raw address strings for NER input
# - To quantify duplicate address strings in a persons register context
# - To create lightweight structural features for exploratory analysis
# - To inspect edge cases that inform annotation guideline design
# - To produce a variability-aware sample of 3000 unique addresses for annotation

#The outputs directly inform:
# - Annotation guideline development
# - NER input representation
# - Sampling design
# - Later model training and evaluation

#-----------------------------------------------------------------------------------------------------------------
#Importing libraries and loading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

#Path to the original tab-delimited address dataset
path = r"C:\Users\valen\Desktop\MSc Dissertation\Code\Data\Address_Data.txt"
df = pd.read_csv(
    path,
    sep="\t",
    quotechar='"',
    encoding="latin1",
    engine="python"
)
df.head()
#-----------------------------------------------------------------------------------------------------------------
#PHASE 1: Profile Original Address Columns
#To examine the completeness and usefulness of address-related fields.
#The aim is to identify which fields should contribute to the raw address string and which fields may introduce
#noise or sparsity into the NER input.

#Address Columns
address_cols = [
    "propertyNumber",
    "propertyName",
    "propertySubdivision",
    "streetPrefix",
    "streetName",
    "streetCouncil",
    "postalCode",
    "island",
    "addressClass",
    "country"
]

address_cols = [c for c in address_cols if c in df.columns]

#Rows and Columns
print("Rows, Columns:", df.shape)

#Missingness and Sparsity
missing_pct = (
    df[address_cols]
    .apply(lambda col: col.isna().mean()*100)
    .round(2)
    .to_frame("missing_percentage")
    .sort_values(by="missing_percentage", ascending=False)
)
missing_pct

#Interpretation:
#propertySubdivision is expected to have high missingness because not every property has a subdivision.
#postalCode has high missingness, which may be problematic for models that rely on postal codes or location-based features.


#Investigating Postal Codes
#Postal codes are checked separately because they may appear in standard, non-standard, missing, or invalid formats.

#Standard format: exactly 3 letters + space + 4 digits
standard_pat = re.compile(r"^[A-Z]{3}\s\d{4}$")

#Usable format: 3 letters + optional space + 4 digits, case-insensitive
usable_pat = re.compile(r"^[A-Z]{3}\s?\d{4}$", re.IGNORECASE)

def postcode_status(x):
    if pd.isna(x):
        return "missing"
    s = str(x).strip()
    if s == "":
        return "missing"

    #Keep original spacing for classification, but compare case-insensitively
    if standard_pat.match(s.upper()):
        return "valid_standard"
    if usable_pat.match(s):
        return "usable_nonstandard"
    return "invalid"

postcode_breakdown = (
    df["postalCode"]
    .apply(postcode_status)
    .value_counts(normalize=True) * 100
).round(2)

postcode_breakdown

#Decision:
#To exclude postalCode from modelling as 91.39% of the entries are either missing or invalid.
#Including this field would introduce sparse and unreliable signals into the NER input representation.


#Investigating Address Class
#addressClass is assessed separately because it is categorical and may contain 
#non-informative values such as missing or "NOT KNOWN".

#Normalising representations of missing values for counting
def normalise_class(x):
    if pd.isna(x):
        return "missing"
    s = str(x).strip().upper()
    if s == "":
        return "missing"
    return s


#Count values and percentages
addressclass_summary = (
    df["addressClass"]
    .apply(normalise_class)
    .value_counts(normalize=True) * 100
).round(2)

addressclass_summary

#Decision:
#To exclude addressClass from modelling as 63.04% of the entries are missing or marked as "NOT KNOWN".
#Given its categorical nature and limited lexical variability, addressClass is not included in the NER input representation.
#-----------------------------------------------------------------------------------------------------------------
#PHASE 2: Construct Raw Address Strings (FOR NER INPUT)
#Constructing the text strings that will later be annotated for NER.
#Only address components considered useful and sufficiently reliable are used.

#Excluded fields:
# - postalCode, due to extreme missingness and invalid formatting
# - addressClass, due to missingness and low lexical usefulness
# - country, because it does not provide useful address-entity variation


#Columns retained for raw address string construction
string_cols = [
    "propertyNumber",
    "propertyName",
    "propertySubdivision",
    "streetPrefix",
    "streetName",
    "streetCouncil",
    "island"
]
string_cols = [c for c in string_cols if c in df.columns]


#Building raw address strings
#This function:
# - skips missing values
# - strips leading and trailing whitespace
# - skips blank strings
# - joins remaining components with commas
#Performs formatting only. Does not normalise, correct, or standardise address content.
#Goal is preserve the original address form for annotation.
def build_raw_address(row, cols):
    parts = []
    for c in cols:
        val = row[c]
        if pd.notna(val):
            val = str(val).strip()
            if val !="" and val.lower() != "nan":
                parts.append(val)
    return ", ".join(parts)

df["raw_address_string"] = df.apply(
    build_raw_address,
    axis=1,
    cols=string_cols
)

df["raw_address_string"].head()
#-----------------------------------------------------------------------------------------------------------------
#PHASE 3: Duplicate Analysis (Persons Register Context)
#Quantifying the repeated rows/addresses

s = df["raw_address_string"].astype(str)

n_total = len(s)
n_unique = s.nunique()
duplicate_rate = (1 - n_unique / n_total) * 100

n_total, n_unique, duplicate_rate


#Interpretation:
#High duplication is expected because multiple persons may reside at the same address.

#Decision:
#Duplicates are retained in the full dataset to preserve the real distribution of addresses in the persons register.
#However, annotation will later operate on unique address strings only, to avoid redundant annotation of identical strings.

s.value_counts().head(10)
#-----------------------------------------------------------------------------------------------------------------
#PHASE 4: Structural Variability Feature Extension
#Splitting address string into tokens (tokens are alphabetic sequences and numeric sequences)
#These features are not intended as final NER model features.
#Instead, they are used for:
# - exploratory analysis
# - detecting edge cases
# - supporting variability-aware sampling
# - informing annotation guideline design


def tokenize(addr):
    return re.findall(r"[A-Za-zÀ-ÿĊċĠġĦħŻż]+|\d+", addr) #roughly splitting the address into chunks

#Computing simple structural features for one address string
#Features to describe variability, not correctness
# - length and token counts
# - digit and punctuation density
# - token repetition (unique token ratio)
# - presence of punctuation edge cases
# - presence of Maltese or accented characters
def compute_features(addr):

    #Handle missing values
    if pd.isna(addr):
        addr = ""
    addr = str(addr)

    #Tokenise address
    tokens = tokenize(addr)

    #Basic size measures
    n_chars = len(addr)     #Total characters
    n_tokens = len(tokens)  #Total tokens

    #Count numeric and punctuation characters
    n_digits = sum(ch.isdigit() for ch in addr)
    n_punct = sum(ch in ",.;:/\\-()[]" for ch in addr)

    #Measure repetition:
    #Low unique_token_ratio = many repeated tokens
    token_lower = [t.lower() for t in tokens]
    unique_token_ratio = len(set(token_lower)) / n_tokens if n_tokens else 0.0

    return {

        #Length and structure
        "n_chars": n_chars,
        "n_tokens": n_tokens,

        #Density measures
        "digit_ratio": n_digits / n_chars if n_chars else 0.0,
        "punct_ratio": n_punct / n_chars if n_chars else 0.0,

        #Repetition/diversity
        "unique_token_ratio": unique_token_ratio,

        #Binary indicators for common edge cases
        "has_slash": int("/" in addr or "\\" in addr),
        "has_hyphen": int("-" in addr),
        "has_parentheses": int("(" in addr or ")" in addr),

        #Presence of non-ASCII/Maltese characters
        "has_diacritics": int(bool(re.search(r"[ĊċĠġĦħŻżÀ-ÿ]", addr)))
    }

#Applying feature extraction to every raw address string
features = df["raw_address_string"].apply(compute_features).apply(pd.Series)

#Combining original data with DEPA features
df_depa = pd.concat([df, features], axis=1)

df_depa[
    ["raw_address_string", "n_chars", "n_tokens", "digit_ratio", "punct_ratio"]
].head(5)
#-----------------------------------------------------------------------------------------------------------------
#PHASE 5: VISUAL EXPLORATION
#The plots help identify unusual patterns that may require special annotation rules or preprocessing decisions.

#Distribution of address length in characters
plt.hist(df_depa["n_chars"], bins=50)
plt.title("Distribution of address string length")
plt.xlabel("Number of characters")
plt.ylabel("Frequency")
plt.show()

#Investigate the most common string lengths.
#This identifies whether unusual spikes are caused by repeated placeholder-like address strings.
df["raw_address_string"].str.len().value_counts().head(10)


#Check the number of rows where the raw address string is exactly "MALTA".
#This explains the large number of 5-character address strings.
(df["raw_address_string"] == "MALTA").sum()


#Distribution of token counts
plt.hist(df_depa["n_tokens"], bins=40)
plt.title("Distribution of token count")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.show()

#Inspecting empty raw address strings
df["raw_address_string"].str.len().value_counts().head(10)
df[df["raw_address_string"].str.len() == 0][string_cols].head(20)



#Distribution of Numeric Density
plt.hist(df_depa["digit_ratio"], bins=30)
plt.title("Digit density in address strings")
plt.xlabel("Digit ratio")
plt.ylabel("Frequency")
plt.show()
#-----------------------------------------------------------------------------------------------------------------
#PHASE 6: EDGE-CASE EXTRACTION
#To find real examples that require explicit annotation rules
#These examples directly inform:
# - boundary decisions
# - handling of numbers
# - punctuation
# - repeated components

def show_examples(condition, n=15, seed=42):

    #Randomly sampling address strings that satisfy a given condition
    #Used to inspect edge cases for annotation guideline design

    subset = df_depa.loc[condition, "raw_address_string"]

    #Avoid errors if there are fewer than n matches
    n_take = min(n, len(subset))

    return subset.sample(
        n=n_take,
        random_state=seed
    ).tolist()


#Very long addresses (top 5)
long_cases = show_examples(
    df_depa["n_chars"] >= df_depa["n_chars"].quantile(0.95)
)


#Addresses with unusally high numeric content
digit_cases = show_examples(
    df_depa["digit_ratio"] >= df_depa["digit_ratio"].quantile(0.95)
)


#Addresses containing slashes
slash_cases = show_examples(df_depa["has_slash"] == 1)

long_cases[:5], digit_cases[:5], slash_cases[:5]
#-----------------------------------------------------------------------------------------------------------------
#PHASE 7: SAVING DEPA OUTPUTS
#Locking decisions before annotation begins
# - no changes to raw_address_string,
# - no changes to included columns,
# - no changes to DEPA logic.

#Save missingness and sparsity results
#missing_pct.to_csv(r"C:\Users\valen\Desktop\MSc Dissertation\Code\Outputs\depa_missing_percentages.csv")

#Save full DEPA dataframe with structural features
#df_depa.to_csv(r"C:\Users\valen\Desktop\MSc Dissertation\Code\Outputs\depa_with_features.csv", index=False)
#-----------------------------------------------------------------------------------------------------------------
#PHASE 8: SAMPLE 3000 UNIQUE ADDRESSES (VARIABILITY-AWARE)
#Sampling design:
# - 2400 addresses from variability-based strata (length, digit density, punctuation)
# - 300 high-frequency addresses (to cover common patterns)
# - 300 random addresses from the remaining pool (to avoid over-structuring)


#Building a unique address table with frequency
#Basic cleaning to remove missing and blank strings
dfu = df_depa.dropna(subset=["raw_address_string"]).copy()
dfu["raw_address_string"] = dfu["raw_address_string"].astype(str).str.strip()
dfu = dfu[dfu["raw_address_string"] != ""]

#Collapse person-level duplicates into unique address strings
#Keep DEPA features by taking the first row per unique string
unique = (dfu.groupby("raw_address_string", as_index=False)
       .agg(
           source_id=("Index", "min"),     #trace-back id
           freq=("raw_address_string", "size"),
           n_chars=("n_chars", "first"),
           digit_ratio=("digit_ratio", "first"),
           has_slash=("has_slash", "first"),
           has_hyphen=("has_hyphen", "first"),
           has_parentheses=("has_parentheses", "first"),
       )
)

print("Person rows:", len(dfu))
print("Unique address strings:", len(unique))
print("Max frequency:", unique["freq"].max())
unique.head()

#Remarks:
# - Data is heavily duplicated at the address level.
# - One address string appears over 600k times (combination of NAs).
# - Sampling from person rows would be statistically meaningless for NER.


#Defining sample sizes
N_TOTAL = 3000
N_STRATA = 2400
N_FREQ = 300
N_RANDOM = 300

rng = np.random.default_rng(123)
chosen = set()

def sample_excluding(df_pool, n, chosen_set, seed):
    df_pool = df_pool[~df_pool["raw_address_string"].isin(chosen_set)]
    n_take = min(n, len(df_pool))
    if n_take <= 0:
        return df_pool.head(0)
    return df_pool.sample(n=n_take, random_state=seed)



#8.1 Stratified Sampling (variability-based)
#Strata are defined using:
# - address length deciles
# - digit density deciles
# - punctuation group, distinguishing addresses with punctuation edge cases

u = unique.copy()

#Creating decile bins
u["len_bin"] = pd.qcut(u["n_chars"].rank(method="first"), 10, labels=False)
u["digit_bin"] = pd.qcut(u["digit_ratio"].rank(method="first"), 10, labels=False)

#Punctuation group (any punctuation indicator)
u["punct_group"] = ((u["has_slash"] == 1) | (u["has_hyphen"] == 1) | (u["has_parentheses"] == 1)).astype(int)

#Stratum key
u["stratum"] = u["len_bin"].astype(str) + "_" + u["digit_bin"].astype(str) + "_" + u["punct_group"].astype(str)

strata = u["stratum"].value_counts().sort_index()
print("Number of strata:", len(strata))
print("Smallest stratum size:", strata.min())
print("Largest stratum size:", strata.max())

#Sampling roughly evenly from each stratum
per_stratum = max(1, N_STRATA // len(strata))

parts = []
for i, (s, count) in enumerate(strata.items()):
    pool = u[u["stratum"] == s]
    take = min(per_stratum, len(pool))
    samp = sample_excluding(pool, take, chosen, seed = 321 + i)
    parts.append(samp)
    chosen.update(samp["raw_address_string"].tolist())

strata_sample = pd.concat(parts, axis = 0).drop_duplicates(subset = ["raw_address_string"])
print("Strata sample size:", len(strata_sample))

#Note:
#The stratified component may be slightly below 2400 because some strata are small. 
#The final sample is topped up later.

#8.2 High-frequency (Common patterns)
freq_pool = u.sort_values("freq", ascending=False)
freq_sample = sample_excluding(freq_pool, N_FREQ, chosen, seed = 999)
chosen.update(freq_sample["raw_address_string"].tolist())

print("Freq sample size:", len(freq_sample))


#8.3 Random (On the remaining)
rand_sample = sample_excluding(u, N_RANDOM, chosen, seed = 4300)
chosen.update(rand_sample["raw_address_string"].tolist())

print("Random sample size:", len(rand_sample))


#8.4 Combining and topping up to exactly 3000
sample_3000 = pd.concat([strata_sample, freq_sample, rand_sample], axis=0)
sample_3000 = sample_3000.drop_duplicates(subset=["raw_address_string"]).copy()

#Topping up to achieve 3000
if len(sample_3000) < N_TOTAL:
    needed = N_TOTAL - len(sample_3000)
    topup = sample_excluding(u, needed, set(sample_3000["raw_address_string"]), seed = 2026)
    sample_3000 = pd.concat([sample_3000, topup], axis = 0).drop_duplicates(subset=["raw_address_string"])

#Trimming if overshot
if len(sample_3000) > N_TOTAL:
    sample_3000 = sample_3000.sample(n=N_TOTAL, random_state=2027)

print("Final sample size:", len(sample_3000))
#-----------------------------------------------------------------------------------------------------------------
#PHASE 9: EXPORT ANNOTATION SAMPLE
#Exporting CSV for Label Studio import
out = sample_3000[[
    "source_id",
    "raw_address_string",
    "freq",
    "n_chars",
    "digit_ratio",
    "has_slash",
    "has_hyphen",
    "has_parentheses",
]].copy()


#out.to_csv(r"C:\Users\valen\Desktop\MSc Dissertation\Code\Outputs\sample_3000_for_annotation.csv",
#           index=False, encoding="utf-8")

#Validation checks for the exported annotation sample
print(out["raw_address_string"].nunique())  #should be 3000
print(out["freq"].describe())
print(out["has_slash"].mean(), out["has_hyphen"].mean(), out["has_parentheses"].mean())


#Exporting full file with raw address string
#df.to_csv(r"C:\Users\valen\Desktop\MSc Dissertation\Code\FINAL\Outputs\full_data_inc_raw_addresses.csv",
#           index=False, encoding="utf-8")