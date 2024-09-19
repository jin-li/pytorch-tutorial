import os
import requests

def download_file(url, filename, directory='.', overwrite=False):
    """
    Download a file from a given URL and save it in a specified directory.

    Args:
        url (str): The URL of the file to be downloaded.
        filename (str): The name of the file to be saved.
        directory (str, optional): The directory where the file will be saved. Defaults to '.'.
        overwrite (bool, optional): Whether to overwrite an existing file with the same name. Defaults to False.

    Returns:
        str: Path to the downloaded file.
    """
    # Check if the directory exists and create it if not
    os.makedirs(directory, exist_ok=True)

    # Construct the full path of the file
    filepath = os.path.join(directory, filename)

    # Check if a file with the same name already exists
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"A file named '{filename}' already exists in '{directory}'. Use 'overwrite=True' to replace it.")

    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file
        with open(filepath, 'wb') as f:
            f.write(response.content)

        return filepath

    except Exception as e:
        raise Exception(f"Failed to download or save the file: {e}")

# URLs and filenames for the WikiText-2 dataset raw data
# See https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
url = [
    "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
    "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
]
filename = [
    "train.txt",
    "valid.txt",
    "test.txt"
]
directory = "../data/wikitext-2"

try:
    for i in range(len(url)):
        download_file(url[i], filename[i], directory, overwrite=True)
    print(f'WikiText-2 dataset downloaded successfully to {directory}.')
except Exception as e:
    print(f'WikiText-2 dataset download failed: {e}')