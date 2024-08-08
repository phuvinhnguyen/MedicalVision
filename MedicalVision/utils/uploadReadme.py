from huggingface_hub import upload_file, HfApi

def replace_readme_in_hf_repo(
        local_readme_path,
        repo_id,
        hf_token,
        commit_message="Update README file"):
    """
    Replace the README.md file in a Hugging Face repository.

    Args:
        local_readme_path (str): Path to the local README.md file.
        repo_id (str): The repository identifier in the format "username/repo_name".
        hf_token (str): Your Hugging Face authentication token.
        commit_message (str): Commit message for the update. Default is "Update README file".
    """
    try:
        # Authenticate using the provided token
        api = HfApi(token=hf_token)
        
        # Upload the README.md file to the repository
        upload_file(
            path_or_fileobj=local_readme_path,
            path_in_repo="README.md",  # Specify the path in the repository
            repo_id=repo_id,
            commit_message=commit_message,
            token=hf_token  # Use the provided token
        )
        print(f"README.md has been successfully updated in the repository '{repo_id}'.")
    except Exception as e:
        print(f"An error occurred while updating the README.md file: {e}")