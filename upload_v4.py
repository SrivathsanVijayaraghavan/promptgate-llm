from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="models/intent_classifier",
    repo_id="srivathsan-vijayaraghavan/promptgate-intent-classifier",
    commit_message="v4: multilingual training data — fix German injection + persona misses, 0 regressions on deepset eval"
)
print("Done")
