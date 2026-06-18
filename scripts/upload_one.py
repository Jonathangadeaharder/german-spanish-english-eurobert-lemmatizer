import os, sys, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import HfApi
lang=sys.argv[1]
api=HfApi()
src=f"onnx/eurobert-lemma-{lang}-210m/model.int8.onnx"
repo=f"Jonandrop/eurobert-lemma-{lang}-210m"
for attempt in range(1,9):
    try:
        print(f"[{lang}] attempt {attempt}", flush=True)
        api.upload_file(path_or_fileobj=src, path_in_repo="model.int8.onnx",
                        repo_id=repo, repo_type="model")
        print(f"[{lang}] DONE", flush=True); break
    except Exception as e:
        print(f"[{lang}] fail: {type(e).__name__}: {str(e)[:160]}", flush=True)
        time.sleep(5)
else:
    sys.exit(1)
