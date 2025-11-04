import logging
from dotenv import load_dotenv
from typing import Literal, Dict, List

load_dotenv()

# logging level
LOG_LEVEL = logging.ERROR

# -------------Simulate--------------
# qdrant setup
QDRANT_ICD_PROCEDURE_COLLECTION_NAME: str = "mimic_iv_icd_codes_procedures"
QDRANT_ICD_PROCEDURE_EMBEDDING_MODEL: str = "jinaai/jina-embeddings-v3"
EMBEDDING_MODEL_DIR = "/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/checkpoints/jina"
QDRANT_URL: str = "http://localhost:6333"


DATASET_NAMES: list[str] = [
    # "appendicitis",
    # "cholecystitis", 
    # "diverticulitis", 
    # "pancreatitis", 
    "pneumonia", 
    # "pulmonary_embolism", 
    # "uti"
]
MAX_SAMPLES: int | None = 1 #None  # if set to None, we use all samples
MAX_EVAL_PARALLEL_SIZE: int | None = (
    1  # maximum number of parallel evaluations (not in interactive mode)
)

# med assistant settings
MEDICAL_ASSISTANT_MODEL: str = "litellm/openai/GPT-OSS-120B" #"litellm/openai/GLM-4.5-Air-FP8" #"gpt-4o" #"gpt-4.1" #"litellm/openai/GLM-4.5V"  #"litellm/openai/llama-3.1-8b-instruct-q8" #"litellm/openai/llama-3.3-70b-instruct-awq"
 #"litellm/openai/Qwen3-235B-A22B-Instruct-2507-FP8" #"Kimi-K2-Instruct" #"Llama-3.3-70B-Instruct" #"medgemma-4b-it-q8" #"meta-llama-3.1-8b-instruct-q8" #"Qwen3-235B-A22B-FP8" #"llama-3.3-70b-instruct-q4km" #"Qwen2.5-VL-72B-Instruct" #"Llama-4-Scout-17B-16E-Instruct" #"deepseek-r1-distill-qwen-32b-q8"
MEDICAL_ASSISTANT_TEMPERATURE: float = 0.01

# patient assistant settings
PATIENT_ASSISTANT_MODEL: str = "litellm/openai/GPT-OSS-120B"
PATIENT_ASSISTANT_TEMPERATURE: float = 0.01


AIDOC_HADM_IDS: Dict[str, List[int]] = {
    "appendicitis": [
        21520386,
    ],
    "cholecystitis": [
        21210624,
    ],
    "diverticulitis": [
        29381122,
    ],
    "pancreatitis": [
        28079494,
    ],
    "pneumonia": [
        28519937,
    ],
    "pulmonary_embolism": [
        26406913,
    ],
    "uti": [
        26514952,
    ],
}

OUTPUT_DIR: str = "./data/formatted_outputs"

#--------------------Evaluate----------------
# patient assistant settings
EVALUATOR_MODEL: str = "medgemma-27b-it-q6" #"Qwen3-235B-A22B-Thinking-2507-FP8" #"GLM-4.5-Air-FP8" #"gpt-4o" #"Llama-4-Maverick-17B-128E-Instruct-FP8" #"Llama-3.3-70B-Instruct" #


