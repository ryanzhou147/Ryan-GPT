"""
Pre-tokenization module for text files.
Implements parallel pre-tokenization with special token handling.
"""
import os
import regex as re
from multiprocessing import Pool
from typing import BinaryIO, Dict

# Default number of worker processes used when splitting files
NUM_PRETOKENIZING_PROCESSES = 5

# Module-level globals for workers
_worker_pat: re.Pattern | None = None
_worker_special_tokens: list[str] | None = None
_worker_initialized = False

def _init_worker(pat_str: str, special_tokens: list[str]) -> None:
    """Initializer run once in each worker process: compile pattern and store tokens."""
    global _worker_pat, _worker_special_tokens, _worker_initialized
    if _worker_initialized:
        return
    _worker_pat = re.compile(pat_str)
    _worker_special_tokens = special_tokens
    _worker_initialized = True


def _build_split_regex(special_tokens: list[str]) -> re.Pattern | None:
    """Build regex to split on special tokens."""
    if not special_tokens:
        return None
    escaped = [re.escape(t) for t in special_tokens]
    return re.compile("|".join(escaped))


def _decode_incremental(carry_bytes: bytes) -> tuple[str, bytes]:
    """Try to decode bytes; return decoded string and remaining tail bytes.
    
    If a UnicodeDecodeError occurs due to incomplete multibyte sequence,
    keep the tail bytes for the next iteration.
    """
    try:
        return carry_bytes.decode("utf-8"), b""
    except UnicodeDecodeError as e:
        decoded = carry_bytes[:e.start].decode("utf-8", errors="ignore")
        tail_bytes = carry_bytes[e.start:]
        return decoded, tail_bytes


def _process_text_segment(text: str, pat: re.Pattern, local_counts: Dict[str, int]) -> None:
    """Apply PAT regex to text segment and update token counts."""
    for match in pat.finditer(text):
        token = match.group()
        local_counts[token] = local_counts.get(token, 0) + 1


def _process_chunk_worker(start: int, end: int, filepath: str) -> Dict[str, int]:
    """Module-level worker: tokenize a byte range of a file.

    Streams the requested range in <=1MiB blocks with incremental UTF-8 decoding
    and handles special token splitting.
    """
    assert _worker_pat is not None and _worker_special_tokens is not None, "Worker not initialized"
    local_counts: Dict[str, int] = {}

    with open(filepath, "rb") as f:
        f.seek(start)
        bytes_to_read = end - start
        READ_BLOCK = 1024 * 1024
        TAIL_CHARS = 4096

        carry_bytes = b""
        split_regex = _build_split_regex(_worker_special_tokens)

        while bytes_to_read > 0:
            this_read = min(READ_BLOCK, bytes_to_read)
            part = f.read(this_read)
            if not part:
                break
            bytes_to_read -= len(part)

            carry_bytes += part

            # Decode incrementally
            decoded, carry_bytes = _decode_incremental(carry_bytes)

            # Process decoded text with special token splitting
            if split_regex:
                pieces = split_regex.split(decoded)
                for sub in pieces[:-1]:
                    _process_text_segment(sub, _worker_pat, local_counts)
                last_fragment = pieces[-1]
                carry_bytes = last_fragment.encode("utf-8") + carry_bytes
            else:
                if len(decoded) > TAIL_CHARS:
                    safe_portion = decoded[:-TAIL_CHARS]
                    _process_text_segment(safe_portion, _worker_pat, local_counts)
                    remaining_fragment = decoded[-TAIL_CHARS:]
                    carry_bytes = remaining_fragment.encode("utf-8") + carry_bytes
                else:
                    carry_bytes = decoded.encode("utf-8") + carry_bytes

        # Flush remaining bytes
        if carry_bytes:
            final_decoded = carry_bytes.decode("utf-8", errors="ignore")
        else:
            final_decoded = ""

        if final_decoded:
            if split_regex:
                pieces = split_regex.split(final_decoded)
                for sub in pieces:
                    _process_text_segment(sub, _worker_pat, local_counts)
            else:
                _process_text_segment(final_decoded, _worker_pat, local_counts)

    return local_counts


class PreTokenizer:
    """Pre-tokenizer using GPT-2 PAT with support for special tokens."""

    def __init__(self, special_tokens: list[str]) -> None:
        self.special_tokens = special_tokens
        self.PAT = self._get_pat()
        self.global_pretokenization_dict: Dict[str, int] = {}
        self.pretokenization_dict_to_bytes: Dict[tuple[bytes], int] = {}

    @staticmethod
    def _get_pat() -> re.Pattern:
        """Return the canonical GPT-2 pretokenization pattern."""
        pat_str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        return re.compile(pat_str)

    def _find_chunk_boundaries(self, file: BinaryIO, num_chunks: int) -> list[int]:
        """Find chunk boundaries by searching for special tokens in the file.
        
        Returns a sorted list of byte offsets where chunks should start/end.
        """
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # Initial uniformly-spaced boundaries
        chunk_size = file_size // num_chunks
        boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        boundaries[-1] = file_size

        # Snap internal boundaries to special token locations
        mini_chunk_size = 4096
        for bi in range(1, len(boundaries) - 1):
            pos = boundaries[bi]
            file.seek(pos)
            while pos < file_size:
                chunk = file.read(mini_chunk_size)
                if not chunk:
                    boundaries[bi] = file_size
                    break
                found_at = chunk.find(b"<|endoftext|>")
                if found_at != -1:
                    boundaries[bi] = pos + found_at
                    break
                pos += mini_chunk_size

        return sorted(set(boundaries))

    def _merge_chunk_results(self, chunk_results: list[Dict[str, int]]) -> None:
        """Merge frequency dicts from all chunks into global dict."""
        for chunk_result in chunk_results:
            for token, count in chunk_result.items():
                self.global_pretokenization_dict[token] = (
                    self.global_pretokenization_dict.get(token, 0) + count
                )

    def _convert_to_bytes_dict(self) -> None:
        """Convert string tokens to tuple-of-bytes format."""
        self.pretokenization_dict_to_bytes = {}
        for token, count in self.global_pretokenization_dict.items():
            token_bytes = token.encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            self.pretokenization_dict_to_bytes[token_tuple] = (
                self.pretokenization_dict_to_bytes.get(token_tuple, 0) + count
            )

    def _create_work_chunks(self, boundaries: list[int], filepath: str) -> list[tuple]:
        """Create worker task tuples from boundaries."""
        return [(boundaries[i], boundaries[i + 1], filepath) for i in range(len(boundaries) - 1)]

    def pretokenize_file_parallel(self, file_path_from_root_folder: str) -> Dict[tuple[bytes], int]:
        """Pretokenize a file in parallel, returning byte-tuple frequency dict."""
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, file_path_from_root_folder)

        with open(file_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, NUM_PRETOKENIZING_PROCESSES)

        work_chunks = self._create_work_chunks(boundaries, file_path)

        with Pool(
            processes=NUM_PRETOKENIZING_PROCESSES,
            initializer=_init_worker,
            initargs=(self.PAT.pattern, self.special_tokens),
        ) as pool:
            results = pool.starmap(_process_chunk_worker, work_chunks)

        self._merge_chunk_results(results)
        self._convert_to_bytes_dict()
        return self.pretokenization_dict_to_bytes
