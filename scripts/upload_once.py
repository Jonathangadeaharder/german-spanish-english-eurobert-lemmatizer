import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import HfApi

lang=sys.argv[1]
HfApi().upload_file(path_or_fileobj=f"onnx/eurobert-lemma-{lang}-210m/model.int8.onnx",
    path_in_repo="model.int8.onnx", repo_id=f"Jonandrop/eurobert-lemma-{lang}-210m",
    repo_type="model")
print("DONE")
