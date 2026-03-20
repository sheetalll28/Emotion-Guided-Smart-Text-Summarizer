# Emotion-Aware Speech Summarization System

## Overview
This project is an end-to-end audio-to-text summarization system that generates context-aware summaries by combining semantic relevance, emotion analysis, and confidence scoring.

The system is designed to work efficiently on low-compute environments by using lightweight speech recognition and optimized processing techniques.

---

## Features

- **Speech-to-Text Conversion**  
  Converts audio input into text using Vosk, enabling offline and resource-efficient processing.

- **Custom Punctuation Restoration**  
  Implements a timestamp-based approach to reconstruct sentence boundaries by analyzing pauses in speech.

- **Emotion-Aware Summarization**  
  Generates summaries using a hybrid scoring mechanism based on:
  - Semantic relevance  
  - Emotion weighting  
  - Confidence scores  

- **Efficient Processing**  
  Designed to run on resource-constrained devices with minimal computational overhead.

---

## Technologies Used

### Languages
- Python  

### Libraries and Tools
- Vosk (speech recognition)  
- librosa (audio feature extraction)  
- NumPy, pandas  

### Concepts
- Natural Language Processing (NLP)  
- Speech Processing  
- Deep Neural Networks  
- Text Summarization  

---
