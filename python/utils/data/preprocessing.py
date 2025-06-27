"""
Text Preprocessing Utilities

This module provides comprehensive text preprocessing pipelines with support for
various text processing tasks including tokenization, normalization, and feature extraction.
Designed with extensibility for vision and audio preprocessing when needed.

Features:
- Text normalization and cleaning
- Tokenization with multiple backends
- Language detection and filtering
- Text augmentation strategies
- Configurable preprocessing pipelines
- Batch processing with progress tracking
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import html

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional tokenizer backends
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    # Text cleaning
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = False
    remove_numbers: bool = False
    remove_punctuation: bool = False
    remove_extra_whitespace: bool = True
    
    # Unicode normalization
    unicode_normalize: str = "NFKC"  # None, NFC, NFKC, NFD, NFKD
    
    # Language filtering
    language_filter: Optional[str] = None  # ISO language code
    min_language_confidence: float = 0.8
    
    # Text filtering
    min_length: int = 1
    max_length: int = 1000000
    filter_duplicates: bool = False
    
    # Tokenization
    tokenizer_type: str = "whitespace"  # whitespace, transformers, spacy, nltk
    tokenizer_model: Optional[str] = None
    max_tokens: Optional[int] = None
    
    # Additional processing
    custom_filters: List[Callable[[str], str]] = field(default_factory=list)


class BasePreprocessor(ABC):
    """Base class for text preprocessors."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    @abstractmethod
    def process(self, text: str) -> str:
        """Process a single text."""
        pass
        
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        return [self.process(text) for text in texts]


class TextCleaner(BasePreprocessor):
    """Text cleaning and normalization."""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for efficient processing."""
        self.patterns = {}
        
        # URL pattern
        self.patterns['url'] = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.patterns['email'] = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number pattern (basic)
        self.patterns['phone'] = re.compile(
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        )
        
        # Numbers pattern
        self.patterns['numbers'] = re.compile(r'\b\d+\b')
        
        # Extra whitespace pattern
        self.patterns['whitespace'] = re.compile(r'\s+')
        
        # Punctuation pattern (if needed)
        self.patterns['punctuation'] = re.compile(r'[^\w\s]')
        
    def process(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
            
        # Remove HTML
        if self.config.remove_html:
            text = self._remove_html(text)
            
        # Remove URLs
        if self.config.remove_urls:
            text = self.patterns['url'].sub('', text)
            
        # Remove emails
        if self.config.remove_emails:
            text = self.patterns['email'].sub('', text)
            
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = self.patterns['phone'].sub('', text)
            
        # Remove numbers
        if self.config.remove_numbers:
            text = self.patterns['numbers'].sub('', text)
            
        # Remove punctuation
        if self.config.remove_punctuation:
            text = self.patterns['punctuation'].sub(' ', text)
            
        # Unicode normalization
        if self.config.unicode_normalize:
            text = unicodedata.normalize(self.config.unicode_normalize, text)
            
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
            
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = self.patterns['whitespace'].sub(' ', text)
            text = text.strip()
            
        # Apply custom filters
        for custom_filter in self.config.custom_filters:
            try:
                text = custom_filter(text)
            except Exception as e:
                logger.warning(f"Custom filter failed: {e}")
                
        return text
        
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = html.unescape(text)
        return text


class LanguageFilter(BasePreprocessor):
    """Language detection and filtering."""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self.language_detector = None
        
        if config.language_filter:
            self._init_language_detector()
            
    def _init_language_detector(self):
        """Initialize language detection."""
        try:
            # Try langdetect first
            from langdetect import detect, detect_langs
            self.language_detector = "langdetect"
            self._detect_func = detect
            self._detect_langs_func = detect_langs
            logger.info("Using langdetect for language detection")
        except ImportError:
            try:
                # Fallback to polyglot
                from polyglot.detect import Detector
                self.language_detector = "polyglot"
                logger.info("Using polyglot for language detection")
            except ImportError:
                logger.warning("No language detection library available")
                self.language_detector = None
                
    def process(self, text: str) -> str:
        """Filter text by language."""
        if not self.config.language_filter or not self.language_detector:
            return text
            
        try:
            if self.language_detector == "langdetect":
                detected_langs = self._detect_langs_func(text)
                for lang in detected_langs:
                    if (lang.lang == self.config.language_filter and 
                        lang.prob >= self.config.min_language_confidence):
                        return text
                return ""  # Filter out
                
            elif self.language_detector == "polyglot":
                from polyglot.detect import Detector
                detector = Detector(text)
                if (detector.language.code == self.config.language_filter and
                    detector.language.confidence >= self.config.min_language_confidence):
                    return text
                return ""  # Filter out
                
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            # Return original text if detection fails
            return text
            
        return text


class TextTokenizer(BasePreprocessor):
    """Text tokenization with multiple backends."""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self.tokenizer = None
        self._init_tokenizer()
        
    def _init_tokenizer(self):
        """Initialize tokenizer based on config."""
        if self.config.tokenizer_type == "transformers":
            self._init_transformers_tokenizer()
        elif self.config.tokenizer_type == "spacy":
            self._init_spacy_tokenizer()
        elif self.config.tokenizer_type == "nltk":
            self._init_nltk_tokenizer()
        else:
            # Default to whitespace tokenizer
            self.tokenizer = "whitespace"
            
    def _init_transformers_tokenizer(self):
        """Initialize HuggingFace transformers tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, falling back to whitespace")
            self.tokenizer = "whitespace"
            return
            
        try:
            model_name = self.config.tokenizer_model or "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Initialized transformers tokenizer: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load transformers tokenizer: {e}")
            self.tokenizer = "whitespace"
            
    def _init_spacy_tokenizer(self):
        """Initialize spaCy tokenizer."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, falling back to whitespace")
            self.tokenizer = "whitespace"
            return
            
        try:
            model_name = self.config.tokenizer_model or "en_core_web_sm"
            self.tokenizer = spacy.load(model_name)
            logger.info(f"Initialized spaCy tokenizer: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load spaCy tokenizer: {e}")
            self.tokenizer = "whitespace"
            
    def _init_nltk_tokenizer(self):
        """Initialize NLTK tokenizer."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, falling back to whitespace")
            self.tokenizer = "whitespace"
            return
            
        try:
            import nltk
            # Download required data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            self.tokenizer = "nltk"
            logger.info("Initialized NLTK tokenizer")
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK tokenizer: {e}")
            self.tokenizer = "whitespace"
            
    def process(self, text: str) -> str:
        """Tokenize text (returns space-separated tokens)."""
        tokens = self.tokenize(text)
        return " ".join(tokens)
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text and return list of tokens."""
        if not text:
            return []
            
        if self.tokenizer == "whitespace":
            tokens = text.split()
        elif hasattr(self.tokenizer, 'tokenize'):
            # Transformers tokenizer
            tokens = self.tokenizer.tokenize(text)
        elif hasattr(self.tokenizer, '__call__'):
            # spaCy tokenizer
            doc = self.tokenizer(text)
            tokens = [token.text for token in doc]
        elif self.tokenizer == "nltk":
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
            
        # Apply max tokens limit
        if self.config.max_tokens and len(tokens) > self.config.max_tokens:
            tokens = tokens[:self.config.max_tokens]
            
        return tokens


class TextFilter(BasePreprocessor):
    """Text filtering based on length and other criteria."""
    
    def process(self, text: str) -> str:
        """Filter text based on criteria."""
        if not text:
            return ""
            
        # Length filtering
        if len(text) < self.config.min_length or len(text) > self.config.max_length:
            return ""
            
        return text


class TextPreprocessingPipeline:
    """Complete text preprocessing pipeline."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.processors = []
        self._build_pipeline()
        
    def _build_pipeline(self):
        """Build preprocessing pipeline based on config."""
        # Text cleaning (always first)
        self.processors.append(TextCleaner(self.config))
        
        # Language filtering
        if self.config.language_filter:
            self.processors.append(LanguageFilter(self.config))
            
        # Text filtering
        self.processors.append(TextFilter(self.config))
        
        # Tokenization (if requested and not whitespace)
        if self.config.tokenizer_type != "whitespace":
            self.processors.append(TextTokenizer(self.config))
            
        logger.info(f"Built preprocessing pipeline with {len(self.processors)} stages")
        
    def process(self, text: str) -> str:
        """Process text through the pipeline."""
        result = text
        
        for processor in self.processors:
            result = processor.process(result)
            if not result:  # Empty result, skip remaining processors
                break
                
        return result
        
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        results = []
        
        for text in texts:
            processed = self.process(text)
            results.append(processed)
            
        return results
        
    def process_dataset(self, dataset, text_field: str = "text") -> List[str]:
        """Process a dataset with progress tracking."""
        from ..logging.terminal_display import progress_context
        
        results = []
        
        with progress_context(
            description="Processing dataset",
            total=len(dataset) if hasattr(dataset, '__len__') else 1000,
            progress_type="data_loading"
        ) as progress:
            
            for i, item in enumerate(dataset):
                if isinstance(item, dict):
                    text = item.get(text_field, "")
                else:
                    text = str(item)
                    
                processed = self.process(text)
                if processed:  # Only add non-empty results
                    results.append(processed)
                    
                progress.advance()
                
        logger.info(f"Processed {len(results)} texts from dataset")
        return results


# Placeholder classes for future modality support
class ImagePreprocessor(BasePreprocessor):
    """Image preprocessing (placeholder for future implementation)."""
    
    def process(self, image) -> Any:
        """Process image (placeholder)."""
        logger.info("Image preprocessing - placeholder implementation")
        raise NotImplementedError("Image preprocessing will be implemented when needed")


class AudioPreprocessor(BasePreprocessor):
    """Audio preprocessing (placeholder for future implementation)."""
    
    def process(self, audio) -> Any:
        """Process audio (placeholder)."""
        logger.info("Audio preprocessing - placeholder implementation") 
        raise NotImplementedError("Audio preprocessing will be implemented when needed")


# Convenience functions
def create_text_pipeline(
    lowercase: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    min_length: int = 1,
    max_length: int = 1000000,
    **kwargs
) -> TextPreprocessingPipeline:
    """Create a simple text preprocessing pipeline."""
    config = PreprocessingConfig(
        lowercase=lowercase,
        remove_html=remove_html,
        remove_urls=remove_urls,
        min_length=min_length,
        max_length=max_length,
        **kwargs
    )
    return TextPreprocessingPipeline(config)


def preprocess_text(text: str, **kwargs) -> str:
    """Preprocess a single text with default settings."""
    pipeline = create_text_pipeline(**kwargs)
    return pipeline.process(text)


def preprocess_texts(texts: List[str], **kwargs) -> List[str]:
    """Preprocess a list of texts with default settings."""
    pipeline = create_text_pipeline(**kwargs)
    return pipeline.process_batch(texts)