import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import heapq

# Ensure you have the necessary NLTK data downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def summarize_text_frequency(text, num_sentences=3):
    """
    Generates an extractive summary of the text based on word frequency.

    Args:
        text (str): The input text to summarize.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: The generated summary.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())

    # Calculate word frequencies
    word_frequencies = defaultdict(int)
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_frequencies[word] += 1

    if not word_frequencies: # Handle cases with no significant words
        return ""

    # Normalize word frequencies
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    # Rank sentences based on word frequencies
    sentences = sent_tokenize(text)
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]

    # Get the top N sentences
    summary_sentences = []
    # Use a check to prevent errors if num_sentences is larger than available sentences
    num_sentences_to_extract = min(num_sentences, len(sentences))

    # Get indices of top sentences based on scores
    # We need to preserve the original order of sentences for readability
    sorted_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences_to_extract]
    sorted_sentence_indices.sort() # Sort by original sentence order
    
    for i in sorted_sentence_indices:
        summary_sentences.append(sentences[i])

    return " ".join(summary_sentences)

if __name__ == '__main__':
    example_text = (
        "The product is TERRIBLE! I've been waiting for WEEKS! Nobody responds! "
        "This is UNACCEPTABLE! I want a refund immediately! The only good thing was the packaging. "
        "Despite the long wait, the customer service representative was polite, but unable to resolve the issue. "
        "The packaging was indeed a high point, indicating attention to detail where it mattered least for the core problem. "
        "My patience has run out, and I expect a swift resolution."
    )
    summary = summarize_text_frequency(example_text, num_sentences=2)
    print("Original Text:", example_text)
    print("\nSummary:", summary)
