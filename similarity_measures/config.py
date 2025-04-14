# config.py
import os
from dotenv import load_dotenv
import torch # Import torch early for device detection

# --- General Configuration ---
YAML_FILE_PATH = "similar.yml"
RESULTS_DIR = "results"
SECTION_SCORES_FILE = os.path.join(RESULTS_DIR, "section_scores.json")
AVERAGE_SCORES_FILE = os.path.join(RESULTS_DIR, "average_scores.json")

# --- Device Selection (for Transformers/PyTorch) ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}") # Add print statement for confirmation

# --- Sentence-BERT Models Configuration ---
# Each dict defines a model setup.
# 'method_key': Unique identifier used in results.
# 'model_id': Hugging Face ID for the model weights.
# 'tokenizer_id': Hugging Face ID for the tokenizer. Can be same as model_id.
# 'model_kwargs': Dictionary of special arguments for AutoModel.from_pretrained.
# 'tokenizer_kwargs': Dictionary of special arguments for AutoTokenizer.from_pretrained.
SBERT_CONFIGS = [
    {
        "method_key": "sbert_cosine_mpnet_base_v2", # Unique key for this setup
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "tokenizer_id": "sentence-transformers/all-mpnet-base-v2",
        "model_kwargs": {}, # No special args needed
        "tokenizer_kwargs": {},
    },
    {
        "method_key": "sbert_cosine_nomic_v2_moe",
        "model_id": "nomic-ai/nomic-embed-text-v2-moe",
        "tokenizer_id": "nomic-ai/nomic-embed-text-v2-moe", # Tokenizer from same repo
        "model_kwargs": {"trust_remote_code": True}, # Requires trusting remote code
        "tokenizer_kwargs": {},
    },
    {
        "method_key": "sbert_cosine_nomic_v1_5_bert_tok",
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "tokenizer_id": "bert-base-uncased", # Using a different tokenizer
        "model_kwargs": {"trust_remote_code": True, "safe_serialization": True},
        "tokenizer_kwargs": {},
    },
    # Add more configurations here if needed
]

    # {
    #     "method_key": "sbert_cosine_linq_embed_mistral",
    #     "model_id": "Linq-AI-Research/Linq-Embed-Mistral",
    #     "tokenizer_id": "Linq-AI-Research/Linq-Embed-Mistral",
    #     "model_kwargs": {}, # No special args needed
    #     "tokenizer_kwargs": {},
    # },
# Note: Models with `trust_remote_code=True` might execute code from the model's repository.
# Ensure you trust the source before using them.

# --- LLM Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
LLM_SIMILARITY_PROMPT_TEMPLATE = """
Rate the semantic similarity between the following two thoughts on a scale of 0.0 to 1.0.
0.0 means completely dissimilar in meaning.
1.0 means semantically identical or perfect paraphrases.
Provide ONLY the numerical score (float). Do not add any explanations.

Thought 1:
{thought1}

Thought 2:
{thought2}

Similarity Score (0.0-1.0):
"""

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Store the base model ID here
GEMINI_EMBEDDING_MODEL_ID = "text-embedding-004"
# GEMINI_EMBEDDING_MODEL_ID = "gemini-embedding-exp-03-07"
# We'll add the 'models/' prefix within the scorer when calling the API
GEMINI_TASK_TYPE = "SEMANTIC_SIMILARITY"

# --- Similarity Method Keys (Prefixes/Constants) ---
METHOD_TFIDF = "tfidf_cosine"
METHOD_SBERT_PREFIX = "sbert_cosine" # Used mainly for grouping concept now
METHOD_LLM = "llm_judge_similarity"
METHOD_GEMINI = "gemini_cosine"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)