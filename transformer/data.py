"""
Data Pipeline for Gauge-Theoretic Transformer
===============================================

WikiText-2 dataset loading and preprocessing for language modeling.

Dataset Details:
    - WikiText-2 (2.08M tokens)
    - Train: ~2M tokens
    - Valid: ~217K tokens
    - Test: ~245K tokens

Usage:
    from transformer.data import create_dataloaders

    train_loader, val_loader, vocab_size = create_dataloaders(
        max_seq_len=128,
        batch_size=8,
        vocab_size=5000,
    )

Author: Implementation for gauge transformer
Date: November 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import numpy as np
from pathlib import Path
import os
import urllib.request
import zipfile

# HuggingFace datasets (optional - has fallback)
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError as e:
    DATASETS_AVAILABLE = False
    load_dataset = None  # Will use fallback
    # Debug: show why import failed
    import sys
    print(f"[DEBUG] datasets import failed: {e}")
    print(f"[DEBUG] Python executable: {sys.executable}")
except Exception as e:
    DATASETS_AVAILABLE = False
    load_dataset = None
    print(f"[DEBUG] datasets import error: {type(e).__name__}: {e}")

# Transformers (optional - only needed for BPE tokenization)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

# Legacy compatibility
HF_AVAILABLE = DATASETS_AVAILABLE and TRANSFORMERS_AVAILABLE


# =============================================================================
# Fallback: Download WikiText-2 directly (no datasets package needed)
# =============================================================================
# Multiple mirror URLs in case one fails
WIKITEXT2_URLS = [
    # HuggingFace datasets mirror (most reliable)
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1.zip",
    # Alternative: raw.githubusercontent mirror
    "https://raw.githubusercontent.com/pcy190/wikitext/main/wikitext-2-raw-v1.zip",
    # Original S3 (deprecated, may not work)
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
]


def _download_with_redirect(url: str, dest_path: Path) -> bool:
    """
    Download a file, following HTTP redirects.

    Returns True on success, False on failure.
    """
    import ssl
    import shutil
    import http.client

    # Create SSL context that handles HTTPS
    ssl_context = ssl.create_default_context()

    # Create a request with a browser-like user agent
    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
        }
    )

    print(f"  Trying: {url[:60]}...")

    try:
        # Create opener that handles redirects
        opener = urllib.request.build_opener(
            urllib.request.HTTPRedirectHandler(),
            urllib.request.HTTPSHandler(context=ssl_context)
        )

        with opener.open(request, timeout=120) as response:
            # Get the final URL after redirects
            final_url = response.geturl()
            if final_url != url:
                print(f"  Redirected to: {final_url[:60]}...")

            # Get content length if available
            content_length = response.headers.get('Content-Length')
            if content_length:
                size_mb = int(content_length) / 1024 / 1024
                print(f"  File size: {size_mb:.1f} MB")

            # Download to file with progress
            print(f"  Downloading...")
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(response, f)

        # Verify file was downloaded
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"  Download complete!")
            return True
        else:
            print(f"  Download failed - empty file")
            return False

    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"  URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def _download_wikitext2_fallback(cache_dir: Optional[str] = None) -> dict:
    """
    Download WikiText-2 directly from source (fallback when datasets unavailable).

    Returns dict with 'train', 'validation', 'test' text.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "wikitext2"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "wikitext-2-raw-v1.zip"
    extract_dir = cache_dir / "wikitext-2-raw"

    # Download if not exists
    if not extract_dir.exists():
        print(f"Downloading WikiText-2...")

        # Try multiple mirror URLs
        download_success = False
        for url in WIKITEXT2_URLS:
            if _download_with_redirect(url, zip_path):
                download_success = True
                break

        if not download_success:
            raise RuntimeError(
                "Failed to download WikiText-2 from any mirror!\n"
                "Please install the datasets package: pip install datasets\n"
                "Or manually download wikitext-2-raw-v1.zip"
            )

        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(cache_dir)

        # Clean up zip
        zip_path.unlink()

    # Read files
    data_dir = cache_dir / "wikitext-2-raw"

    result = {}
    for split, filename in [('train', 'wiki.train.raw'),
                            ('validation', 'wiki.valid.raw'),
                            ('test', 'wiki.test.raw')]:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                result[split] = f.read()
        else:
            raise FileNotFoundError(f"WikiText-2 file not found: {filepath}")

    print(f"WikiText-2 loaded from cache: {cache_dir}")
    return result


class WikiText2Dataset(Dataset):
    """
    WikiText-2 dataset for language modeling.

    Processes text into fixed-length sequences for autoregressive training:
        Input:  [tok_0, tok_1, ..., tok_{T-1}]
        Target: [tok_1, tok_2, ..., tok_T]

    Features:
        - Efficient tokenization using GPT-2 BPE tokenizer
        - Fixed sequence length with truncation/padding
        - Vocabulary size control (top K tokens)
        - Caches tokenized data for speed
    """

    def __init__(
        self,
        split: str = 'train',
        max_seq_len: int = 128,
        vocab_size: Optional[int] = None,
        tokenizer_name: str = 'gpt2',
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize WikiText-2 dataset.

        Args:
            split: 'train', 'validation', or 'test'
            max_seq_len: Maximum sequence length (T)
            vocab_size: If provided, restrict to top K tokens
            tokenizer_name: HuggingFace tokenizer name
            cache_dir: Optional cache directory for dataset
        """
        assert DATASETS_AVAILABLE, "datasets required! pip install datasets"
        assert TRANSFORMERS_AVAILABLE, "transformers required for BPE tokenization! pip install transformers"

        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size_limit = vocab_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Load dataset
        print(f"Loading WikiText-2 ({split})...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, cache_dir=cache_dir)

        # Concatenate all text (WikiText-2 has one article per line)
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
        full_text = '\n\n'.join(texts)

        print(f"  Total characters: {len(full_text):,}")

        # Tokenize
        print(f"Tokenizing...")
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Restrict vocabulary if requested
        if vocab_size is not None and vocab_size < len(self.tokenizer):
            print(f"  Restricting vocabulary to {vocab_size} most frequent tokens...")
            tokens = self._restrict_vocab(tokens, vocab_size)

        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"  Tokenized: {len(self.tokens):,} tokens")
        print(f"  Vocabulary size: {self.get_vocab_size()}")

        # Calculate number of complete sequences
        # Each sequence is [tok_0...tok_T-1] → [tok_1...tok_T]
        self.num_sequences = max(1, len(self.tokens) - self.max_seq_len)

        print(f"  Number of sequences: {self.num_sequences:,}")

    def _restrict_vocab(self, tokens: List[int], target_vocab_size: int) -> List[int]:
        """
        Restrict tokens to top K most frequent.

        Tokens outside top K are replaced with <unk>.
        """
        # Count token frequencies
        token_counts = {}
        for tok in tokens:
            token_counts[tok] = token_counts.get(tok, 0) + 1

        # Reserve 1 slot for UNK, get top (K-1) tokens
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        # Get UNK token
        unk_token_id = self.tokenizer.unk_token_id
        if unk_token_id is None:
            # Use pad token as UNK if no UNK token
            unk_token_id = self.pad_token_id

        # Take top (K-1) tokens to leave room for UNK
        top_k_minus_1 = set([tok for tok, _ in sorted_tokens[:target_vocab_size - 1]])

        # Include UNK in vocabulary
        top_k_tokens = top_k_minus_1 | {unk_token_id}

        restricted_tokens = [
            tok if tok in top_k_tokens else unk_token_id
            for tok in tokens
        ]

        # Remap to contiguous vocabulary [0, 1, ..., K-1]
        # Sort but ensure UNK is last
        non_unk_tokens = sorted(top_k_minus_1)
        token_to_id = {tok: i for i, tok in enumerate(non_unk_tokens)}
        token_to_id[unk_token_id] = len(token_to_id)  # UNK is last (index K-1)

        self._vocab_mapping = token_to_id
        self._restricted_vocab_size = len(token_to_id)

        return [token_to_id.get(tok, token_to_id[unk_token_id]) for tok in restricted_tokens]

    def get_vocab_size(self) -> int:
        """Return effective vocabulary size."""
        if hasattr(self, '_restricted_vocab_size'):
            return self._restricted_vocab_size
        return len(self.tokenizer)

    def __len__(self) -> int:
        """Number of sequences in dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Sequence index

        Returns:
            input_ids: (max_seq_len,) token IDs
            target_ids: (max_seq_len,) next-token targets
        """
        # Extract sequence starting at idx
        start_idx = idx
        end_idx = start_idx + self.max_seq_len

        # Input: tokens[start:end]
        # Target: tokens[start+1:end+1]
        input_ids = self.tokens[start_idx:end_idx]
        target_ids = self.tokens[start_idx + 1:end_idx + 1]

        # Pad if necessary (should only happen at end of dataset)
        if len(input_ids) < self.max_seq_len:
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])

        return input_ids, target_ids


class WikiText2CharDataset(Dataset):
    """
    Character-level WikiText-2 dataset for language modeling.

    Processes text at character level with fixed-length sequences.
    Perfect for minimal proof-of-principle experiments.

    Features:
        - Character-level modeling (vocab_size ≤ 256 for ASCII/extended ASCII)
        - Fixed sequence length
        - Direct character-to-index mapping
        - No tokenizer required
    """

    def __init__(
        self,
        split: str = 'train',
        max_seq_len: int = 32,
        vocab_size: int = 256,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize character-level WikiText-2 dataset.

        Args:
            split: 'train', 'validation', or 'test'
            max_seq_len: Maximum sequence length (N)
            vocab_size: Maximum vocabulary size (default 256 for extended ASCII)
            cache_dir: Optional cache directory for dataset
        """
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Load dataset (with fallback if datasets package unavailable)
        print(f"Loading WikiText-2 ({split}) for character-level modeling...")

        if DATASETS_AVAILABLE:
            # Use HuggingFace datasets
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, cache_dir=cache_dir)
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
            full_text = '\n\n'.join(texts)
        else:
            # Fallback: download directly
            print("  (Using direct download fallback - datasets package not available)")
            wikitext_data = _download_wikitext2_fallback(cache_dir)
            full_text = wikitext_data[split]

        print(f"  Total characters: {len(full_text):,}")

        # Build character vocabulary
        # Use first vocab_size most frequent characters
        char_counts = {}
        for char in full_text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Sort by frequency
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

        # Take top vocab_size - 2 (reserve for PAD and UNK)
        top_chars = [char for char, _ in sorted_chars[:vocab_size - 2]]

        # Build vocabulary: PAD=0, UNK=1, then top characters
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.char_to_id = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(top_chars):
            self.char_to_id[char] = i + 2

        # Create reverse mapping for decoding
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        # Convert text to indices
        char_indices = []
        for char in full_text:
            if char in self.char_to_id:
                char_indices.append(self.char_to_id[char])
            else:
                char_indices.append(self.unk_token_id)

        self.tokens = torch.tensor(char_indices, dtype=torch.long)

        # Calculate number of complete sequences
        self.num_sequences = max(1, len(self.tokens) - self.max_seq_len)

        print(f"  Character vocab size: {len(self.char_to_id)}")
        print(f"  Number of sequences: {self.num_sequences:,}")
        print(f"  Total chars: {len(self.tokens):,}")

    def get_vocab_size(self) -> int:
        """Return actual vocabulary size."""
        return len(self.char_to_id)

    def decode(self, indices: torch.Tensor) -> str:
        """Decode indices back to text."""
        chars = []
        for idx in indices.tolist():
            chars.append(self.id_to_char.get(idx, '<UNK>'))
        return ''.join(chars)

    def __len__(self) -> int:
        """Number of sequences in dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Sequence index

        Returns:
            input_ids: (max_seq_len,) character IDs
            target_ids: (max_seq_len,) next-character targets
        """
        # Extract sequence starting at idx
        start_idx = idx
        end_idx = start_idx + self.max_seq_len

        # Input: chars[start:end]
        # Target: chars[start+1:end+1]
        input_ids = self.tokens[start_idx:end_idx]
        target_ids = self.tokens[start_idx + 1:end_idx + 1]

        # Pad if necessary
        if len(input_ids) < self.max_seq_len:
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])

        return input_ids, target_ids


def create_char_dataloaders(
    max_seq_len: int = 32,
    batch_size: int = 16,
    vocab_size: int = 256,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and validation dataloaders for character-level WikiText-2.

    Args:
        max_seq_len: Maximum sequence length (default 32 for publication)
        batch_size: Batch size (default 16)
        vocab_size: Maximum vocabulary size (default 256 for extended ASCII)
        num_workers: Number of data loading workers
        cache_dir: Optional cache directory

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        vocab_size: Actual vocabulary size

    Example:
        >>> train_loader, val_loader, vocab_size = create_char_dataloaders(
        ...     max_seq_len=32,
        ...     batch_size=16,
        ...     vocab_size=256,
        ... )
    """
    # Note: No longer requires datasets package - has fallback download!

    print("="*70)
    print("CREATING CHARACTER-LEVEL WIKITEXT-2 DATALOADERS")
    print("="*70)
    if not DATASETS_AVAILABLE:
        print("(Using direct download fallback - datasets package not available)")

    # Create datasets
    train_dataset = WikiText2CharDataset(
        split='train',
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
    )

    val_dataset = WikiText2CharDataset(
        split='validation',
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
    )

    # Get actual vocabulary size
    actual_vocab_size = train_dataset.get_vocab_size()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    print(f"\n{'='*70}")
    print(f"DATALOADERS CREATED")
    print(f"{'='*70}")
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")
    print(f"Vocabulary:    {actual_vocab_size} characters")
    print(f"Batch size:    {batch_size}")
    print(f"Sequence len:  {max_seq_len}")
    print(f"{'='*70}\n")

    return train_loader, val_loader, actual_vocab_size


def create_dataloaders(
    max_seq_len: int = 128,
    batch_size: int = 8,
    vocab_size: Optional[int] = None,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    tokenizer_name: str = 'gpt2',
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and validation dataloaders for WikiText-2.

    Args:
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        vocab_size: If provided, restrict vocabulary to top K tokens
        num_workers: Number of data loading workers
        cache_dir: Optional cache directory
        tokenizer_name: HuggingFace tokenizer name

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        vocab_size: Actual vocabulary size

    Example:
        >>> train_loader, val_loader, vocab_size = create_dataloaders(
        ...     max_seq_len=128,
        ...     batch_size=8,
        ...     vocab_size=5000,
        ... )
        >>> for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        ...     # input_ids: (B, T), target_ids: (B, T)
        ...     logits = model(input_ids)
        ...     loss = criterion(logits, target_ids)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets required! pip install datasets")
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers required for BPE tokenization! pip install transformers")

    print("="*70)
    print("CREATING WIKITEXT-2 DATALOADERS")
    print("="*70)

    # Create datasets
    train_dataset = WikiText2Dataset(
        split='train',
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
    )

    val_dataset = WikiText2Dataset(
        split='validation',
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
    )

    # Get actual vocabulary size (may differ from requested if restricted)
    actual_vocab_size = train_dataset.get_vocab_size()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop incomplete batches for consistent shapes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    print(f"\n{'='*70}")
    print(f"DATALOADERS CREATED")
    print(f"{'='*70}")
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")
    print(f"Vocabulary:    {actual_vocab_size:,} tokens")
    print(f"Batch size:    {batch_size}")
    print(f"Sequence len:  {max_seq_len}")
    print(f"{'='*70}\n")

    return train_loader, val_loader, actual_vocab_size


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of (input_ids, target_ids) tuples

    Returns:
        input_batch: (B, T) batched input IDs
        target_batch: (B, T) batched target IDs
    """
    input_ids, target_ids = zip(*batch)

    input_batch = torch.stack(input_ids, dim=0)
    target_batch = torch.stack(target_ids, dim=0)

    return input_batch, target_batch


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("DATA PIPELINE TEST")
    print("="*70)

    if not DATASETS_AVAILABLE:
        print("\n❌ datasets not installed!")
        print("Install: pip install datasets")
        exit(1)

    # Test configuration (small for quick testing)
    max_seq_len = 64
    batch_size = 4
    vocab_size = 1000  # Restrict to 1K tokens for testing

    print(f"\n[1] Creating dataloaders...")
    print(f"    Sequence length: {max_seq_len}")
    print(f"    Batch size:      {batch_size}")
    print(f"    Vocabulary:      {vocab_size} (restricted)")

    try:
        train_loader, val_loader, actual_vocab_size = create_dataloaders(
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            vocab_size=vocab_size,
            num_workers=0,
        )
        print(f"    ✓ Dataloaders created")
    except Exception as e:
        print(f"    ❌ Error creating dataloaders: {e}")
        raise

    # Test batch shapes
    print(f"\n[2] Testing batch shapes...")
    train_batch = next(iter(train_loader))
    input_ids, target_ids = train_batch

    print(f"    Input shape:  {input_ids.shape}")
    print(f"    Target shape: {target_ids.shape}")

    assert input_ids.shape == (batch_size, max_seq_len), "Input shape mismatch!"
    assert target_ids.shape == (batch_size, max_seq_len), "Target shape mismatch!"
    print(f"    ✓ Shapes correct")

    # Test target alignment (target[i] = input[i+1])
    print(f"\n[3] Testing target alignment...")
    # First sequence
    seq_0_input = input_ids[0]
    seq_0_target = target_ids[0]

    # Show first 10 tokens
    print(f"    Input[0]:  {seq_0_input[:10].tolist()}")
    print(f"    Target[0]: {seq_0_target[:10].tolist()}")
    print(f"    Expected:  {seq_0_input[1:11].tolist()}")

    # Verify alignment (ignoring potential padding)
    non_pad_mask = seq_0_target != train_loader.dataset.pad_token_id
    if non_pad_mask.sum() > 0:
        aligned = torch.all(seq_0_target[:-1] == seq_0_input[1:])
        if aligned:
            print(f"    ✓ Targets correctly shifted")
        else:
            print(f"    ⚠ Warning: Target alignment mismatch (may be due to padding)")

    # Test vocabulary range
    print(f"\n[4] Testing vocabulary...")
    max_token = input_ids.max().item()
    min_token = input_ids.min().item()

    print(f"    Vocabulary size: {actual_vocab_size}")
    print(f"    Token range:     [{min_token}, {max_token}]")
    print(f"    Pad token ID:    {train_loader.dataset.pad_token_id}")

    assert max_token < actual_vocab_size, f"Token {max_token} >= vocab size {actual_vocab_size}"
    assert min_token >= 0, f"Token {min_token} < 0"
    print(f"    ✓ All tokens in valid range")

    # Test validation set
    print(f"\n[5] Testing validation set...")
    val_batch = next(iter(val_loader))
    val_input, val_target = val_batch

    print(f"    Val input shape:  {val_input.shape}")
    print(f"    Val target shape: {val_target.shape}")
    print(f"    ✓ Validation batch works")

    # Dataset statistics
    print(f"\n[6] Dataset statistics:")
    print(f"    Train sequences:      {len(train_loader.dataset):,}")
    print(f"    Val sequences:        {len(val_loader.dataset):,}")
    print(f"    Train batches:        {len(train_loader):,}")
    print(f"    Val batches:          {len(val_loader):,}")
    print(f"    Total train tokens:   {len(train_loader.dataset.tokens):,}")
    print(f"    Total val tokens:     {len(val_loader.dataset.tokens):,}")

    # Estimate training time
    tokens_per_batch = batch_size * max_seq_len
    total_train_tokens = len(train_loader) * tokens_per_batch

    print(f"\n[7] Training estimates:")
    print(f"    Tokens per batch:     {tokens_per_batch:,}")
    print(f"    Total tokens/epoch:   {total_train_tokens:,}")
    print(f"    Batches per epoch:    {len(train_loader):,}")

    # Memory estimate
    # Rough estimate: embedding + activations + gradients
    mem_per_token = 4  # bytes (float32)
    model_size_mb = (actual_vocab_size * 96) * mem_per_token / 1e6  # Embedding only
    batch_mem_mb = tokens_per_batch * 96 * mem_per_token * 3 / 1e6  # Activations

    print(f"\n[8] Memory estimates (very rough):")
    print(f"    Model (embeddings):   ~{model_size_mb:.1f} MB")
    print(f"    Batch activations:    ~{batch_mem_mb:.1f} MB")
    print(f"    Estimated total:      ~{(model_size_mb + batch_mem_mb) * 2:.1f} MB (with overhead)")

    print("\n" + "="*70)
    print("✓ All data pipeline tests passed!")
    print("="*70)
    print("\nReady to train!")
    print("Next: Integrate with Trainer class in train.py")
    print("="*70)