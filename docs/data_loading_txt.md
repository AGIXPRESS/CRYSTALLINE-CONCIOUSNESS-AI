# Data Loading Process for .txt Files

This document outlines the process of loading .txt files in the `unified_data_loader.py` script.

## Scanning the Directory

The `scan_directory()` function is responsible for identifying all .txt files within the specified data directory. It uses `os.walk()` to recursively traverse the directory and its subdirectories.

```python
def scan_directory(self) -> Dict[str, List[str]]:
    """Scans the directory for .txt files"""
    logger.info(f"Scanning directory: {self.data_dir}")
    for root, _, files in os.walk(self.data_dir):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                self.data_files["txt"].append(file_path)
    logger.info(f"Found {len(self.data_files['txt'])} txt files")
```

This function populates the `self.data_files["txt"]` list with the absolute paths of all discovered .txt files.

## Loading the Data

The `load_data()` method then loads the content of each .txt file. This is performed within a try-except block for robust error handling. File handles are tracked with the `FileHandleTracker` to prevent resource leaks.

```python
def load_data(self, file_types: List[str] = None) -> None:
    """Loads the data from the data files."""
    if not file_types:
        file_types = ["txt"]

    if "txt" in file_types:
        logger.info("Loading text files")
        loaded_texts = []
        for file_path in self.data_files["txt"]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                loaded_texts.append(content)
                logger.debug(f"Loaded txt file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        self.loaded_data["txt"] = loaded_texts
        logger.info(f"Loaded {len(loaded_texts)} txt files")
```

### Expected Format

*   The data loader expects the `.txt` files to contain plain text. There are currently no specific assumptions made about the content's structure or encoding.
*   The files are read using UTF-8 encoding with error replacement to handle potential encoding issues.

### Potential Issues and Limitations

*   **Encoding Issues:** The `errors='replace'` argument in the `open()` function replaces any characters that cannot be decoded with a replacement character. This avoids crashes but may lead to data loss.
*   **File Size:** Large files may cause memory issues. Consider implementing chunked reading for very large files.
*   **System Files:**  The loading process skips common system files (`.DS_Store`, `Thumbs.db`, etc.) to prevent accidental loading of irrelevant data.  See the `SYSTEM_FILES` list in `unified_data_loader.py` for the complete list of skipped files.
*  **File Handles:**  The `_safe_open_file` function makes sure file handles are closed, even in case of errors.  A `FileHandleTracker` logs if there are any lingering open files.

