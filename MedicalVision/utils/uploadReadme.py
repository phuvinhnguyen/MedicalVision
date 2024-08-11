from huggingface_hub import upload_file, HfApi

def create_readme():
    pass

def write_file_in_hf_repo(
        local_path,
        repo_id,
        hf_token,
        commit_message="Update README file",
        revision=None):
    try:
        # Authenticate using the provided token
        api = HfApi(token=hf_token)
        
        # Upload the README.md file to the repository
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo="README.md",  # Specify the path in the repository
            repo_id=repo_id,
            commit_message=commit_message,
            token=hf_token,  # Use the provided token
            revision=revision,
        )
        print(f"README.md has been successfully updated in the repository '{repo_id}'.")
    except Exception as e:
        print(f"An error occurred while updating the README.md file: {e}")