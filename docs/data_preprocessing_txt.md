# Data Preprocessing for .txt Files

This document describes the data preprocessing steps applied to `.txt` files within the `unified_data_loader.py` script. The goal is to prepare the text data for effective model training by cleaning, tokenizing, and creating a numerical representation.

## 1. Text Cleaning (`_clean_text` Function)

The `_clean_text()` function performs several cleaning operations to remove noise and standardize the text.

```python
def _clean_text(self, text: str) -> str:
    """
    Clean and preprocess the text by removing URLs, special characters, and extra whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Convert all text to lowercase
    text = text.lower()
    
    return text
```

This function performs the following steps:

1.  **Remove URLs**: Uses regular expressions to remove any URLs present in the text.
2.  **Remove Special Characters**: Removes all characters that are not alphanumeric or whitespace.
3.  **Remove Extra Whitespace**: Collapses multiple whitespace characters into a single space.
4.  **Convert to Lowercase**: Converts all text to lowercase for uniformity.

These steps ensure that the text is standardized for subsequent tokenization and numerical representation.

### Adjustments

*   To modify URL removal, adjust the regular expression pattern used.
*   To preserve certain special characters, modify the character set in the second `re.sub` operation.
*   For more sophisticated cleaning, consider using libraries like Beautiful Soup or specialized NLP tools.

## 2. Tokenization and Vocabulary Creation (`process_data` Function)

The `process_data()` function handles tokenization and vocabulary creation. This involves splitting the text into words (tokens), filtering infrequent words, and creating a mapping from tokens to numerical indices.

```python
    def process_data(self, min_frequency: int = 5, sequence_length: int = 256, max_vocab_size: int = 50000) -> Union[np.ndarray, None]:
        """Process the loaded data."""
        if not self.loaded_data:
            logger.warning("No data loaded. Call load_data() first.")
            return None

        if "txt" in self.loaded_data and self.loaded_data["txt"]:
            logger.info("Processing text data: Cleaning, tokenizing, vectorizing...")
            # Combine all loaded text data
            all_text_entries = self.loaded_data["txt"]
            all_text = "".join([entry['content'] for entry in all_text_entries])
            
            # --- Text Cleaning and Filtering ---
            cleaned_text = self._clean_text(all_text)  # Clean special chars, URLs, etc.

            # --- Tokenization and Vocabulary Creation ---
            # Create simple token set based on word frequency
            word_counts = {}
            for word in cleaned_text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Filter infrequent words to reduce vocabulary size
            frequent_words = [word for word, count in word_counts.items() if count >= min_frequency]
            
            # Create vocabulary based on frequent tokens only
            vocabulary = sorted(list(set(frequent_words)))[:max_vocab_size]  # Limit vocab size
            
            # Add special tokens
            vocabulary = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + vocabulary  # Special tokens
            
            # Create token-to-index mapping
            token_to_idx = {token: i for i, token in enumerate(vocabulary)}
```

Key steps in this function:

1.  **Combining Text Data**: Combines all text entries from loaded `.txt` files into a single string.
2.  **Word Tokenization**: Splits the cleaned text into individual words using `text.split()`.
3.  **Frequency-Based Filtering**: Counts word occurrences and filters out infrequent words based on `min_frequency`.
4.  **Vocabulary Creation**: Creates a sorted list of unique, frequent words to form the vocabulary, limiting the vocabulary size to `max_vocab_size`.
5.  **Special Tokens**: Adds special tokens like `<PAD>`, `<UNK>`, `<SOS>`, and `<EOS>` to the vocabulary for padding, unknown words, start-of-sequence, and end-of-sequence, respectively.
6.  **Token-to-Index Mapping**: Creates a dictionary mapping each token to a unique numerical index.

### Adjustments

*   Adjust the `min_frequency` parameter to control the minimum word frequency for inclusion in the vocabulary.
*   Modify the `max_vocab_size` parameter to control the maximum size of the vocabulary.
*   Customize the tokenization process by using a different tokenization method (e.g., using regular expressions for more complex token splitting).
*   Use a different library for tokenization, such as spaCy or nltk.
*   Modify the special tokens.

## 3. Numerical Representation of Data

The `process_data()` function then converts the tokenized text into a numerical representation using the created vocabulary mapping.

```python
            # Convert the entire text to indices, replacing unknown tokens with <UNK>
            indices = [token_to_idx.get(word, token_to_idx["<UNK>"]) for word in cleaned_text.split()]
            # convert from list to numpy array for processing
            self.processed_data = np.array(indices, dtype=np.int32)
```

This step replaces each token with its corresponding index from the `token_to_idx` mapping. If a token is not found in the vocabulary, it is replaced with the index of the `<UNK>` token.  The resulting list of indices is converted into a NumPy array.

### Adjustments

*   Modify the data type (`dtype`) of the NumPy array based on the model's requirements.
*   Implement different methods for handling unknown tokens (e.g., using subword tokenization or more sophisticated OOV handling techniques).
*   Use other functions to change how the numpy array is made for more or less memory efficiency.

## Additional Considerations

*   **Memory Usage**: For very large text datasets, consider using techniques like chunking or streaming to avoid loading the entire dataset into memory at once.
*   **Tokenization**: The current tokenization method is very basic and simply splits the text by whitespace. Consider using more advanced tokenization techniques for better results.

This documentation provides a clear explanation of the data preprocessing steps performed on `.txt` files, along with instructions on how to customize these steps for different requirements.

