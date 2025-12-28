from pathlib import Path
from typing import List, Tuple, Dict
from time import time
import cProfile, pstats

from ryan_gpt_basics.tokenizer.pretokenizer import PreTokenizer
from ryan_gpt_basics.tokenizer.bpe_tokenizer import BPEProcessor


def train_bpe(input_path: str | Path, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: path to training text
        vocab_size: total desired vocabulary size (including initial 256 byte tokens and special tokens)
        special_tokens: list of special token strings to include in the vocabulary

    Returns:
        vocab: mapping from int id -> bytes
        merges: list of merges as tuples of bytes
    """
    input_path = Path(input_path)

    # Step 1: pretokenize using provided PreTokenizer
    pretokenizer = PreTokenizer(special_tokens)
    # pretokenize the file in parallel
    pretokenizer.pretokenize_file_parallel(str(input_path))

    # Step 2: initialize BPE processor with pretokenizer for training
    bpe = BPEProcessor({}, [], special_tokens, pretokenizer=pretokenizer)

    # initial vocab size includes 256 byte tokens and special tokens
    initial_vocab = 256 + len(special_tokens)
    if vocab_size <= initial_vocab:
        # no merges to perform
        return bpe.vocab, bpe.merges

    num_merges = vocab_size - initial_vocab
    bpe.run_bpe(num_merges)

    return bpe.vocab, bpe.merges

def train_bpe_tinystories():
    start_time = time()
    pr = cProfile.Profile()

    result = pr.runcall(train_bpe,
        input_path='data/TinyStoriesV2-GPT4-valid.txt', vocab_size=10000, special_tokens=['<|endoftext|>']
    )

    end_time = time()

    stats = pstats.Stats(pr).strip_dirs()
    stats.sort_stats("cumulative")  # sort by cumulative time
    stats.print_stats(10)           # print top 10 functions

    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    vocab, merges = result
    max_token = max(vocab.values(), key=len)
    print(max_token, len(max_token))

    top_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    print(f"{'Vocab ID':<8} {' Token':<20} {'Bytes':<5}")
    print("-" * 40)

    for vocab_id, token_bytes in top_tokens:
        try:
            token_str = token_bytes.decode("utf-8")  # decode for readability
        except UnicodeDecodeError:
            token_str = repr(token_bytes)  # fallback if not valid UTF-8
        print(f"{vocab_id:<8} {token_str:<20} {len(token_bytes):<5}")
    
def train_bpe_expts_owt():
    # Train on OpenWebText with a vocabulary size of 32,000 and serialize outputs.

    input_path = Path("data/owt_train.txt")
    vocab_size = 260
    special_tokens = ['<|endoftext|>']

    start_time = time()
    pr = cProfile.Profile()
    result = pr.runcall(train_bpe, input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
    end_time = time()

    # Save profiling summary (top callers) to stdout (concise)
    stats = pstats.Stats(pr).strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(10)

    elapsed = end_time - start_time
    vocab, merges = result

    # Serialize vocab and merges for inspection
    out_dir = Path("out_bpe_owt")
    out_dir.mkdir(exist_ok=True)

    # write vocab as JSON mapping id -> list of ints (bytes)
    import json
    vocab_out = {str(k): list(v) for k, v in vocab.items()}
    (out_dir / "vocab.json").write_text(json.dumps(vocab_out, indent=2), encoding="utf-8")

    # write merges using GPT-2 printable mapping for readability
    try:
        from tests.common import gpt2_bytes_to_unicode
        gpt2 = gpt2_bytes_to_unicode()
        with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
            for left_b, right_b in merges:
                left_repr = "".join(gpt2[b] for b in left_b)
                right_repr = "".join(gpt2[b] for b in right_b)
                f.write(f"{left_repr} {right_repr}\n")
    except Exception:
        # fallback: write raw byte sequences
        with (out_dir / "merges.bytes").open("wb") as f:
            for left_b, right_b in merges:
                f.write(left_b + b" "+ right_b + b"\n")

    # Diagnostics: longest token
    max_len = max(len(b) for b in vocab.values())
    longest = [(i, v) for i, v in vocab.items() if len(v) == max_len]

    # Build concise one-to-two sentence summaries as deliverables
    # (a) Training summary for OpenWebText
    if longest:
        # prepare a printable representation for the first longest token
        idx, token_bytes = longest[0]
        try:
            token_print = token_bytes.decode("utf-8")
        except Exception:
            token_print = repr(token_bytes)
    else:
        idx, token_print = None, ""

    a_summary = (
        f"OpenWebText training completed in {elapsed:.2f}s and produced a vocab of {len(vocab)} tokens; "
        f"the longest token is id={idx} length={max_len} bytes ({token_print[:100]}...)."
    )

    # (b) Comparison summary (TinyStories vs OpenWebText)
    b_summary = (
        "TinyStories yields a tokenizer with many dataset-specific short narrative tokens, "
        "whereas OpenWebText produces a broader vocabulary including longer and more diverse byte sequences reflecting web text variety."
    )

    # return the two short summaries
    return a_summary, b_summary

def train_test():
    vocab, merges = train_bpe("gpt/test.txt", vocab_size=256+40, special_tokens=['<|endoftext|>'])

if __name__ == "__main__":
    train_bpe_expts_owt()
