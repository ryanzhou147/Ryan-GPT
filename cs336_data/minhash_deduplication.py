# Problem (minhash_deduplication): 8 points
# Write a function that takes a list of paths to input files and performs fuzzy document deduplication
# with minhash and LSH. In particular, your function should compute minhash signatures for each
# document in the provided list of paths, use LSH with the provided number of bands to identify candidate
# duplicates, and then compute the true ngram Jaccard similarity between candidate duplicates and
# remove those that exceed a given threshold. To improve recall (following Penedo et al., 2023), normalize
# the text before computing minhash signatures and/or comparing Jaccard similarity by lowercasing,
# removing punctuation, normalizing whitespaces, and removing accents, and applying NFD unicode
# normalization.
# Deliverable: A function that performs fuzzy document deduplication. Your function should take
# at least the following arguments: (a) a list of paths to its input files, (b) the number of hashes to use
# for computing minhash signatures, (c) the number of bands to use for LSH, (d) the n-gram length
# (in words) for computing minhash signatures, and (e) an output directory. You may assume that the
# number of hashes to use for computing minhash signatures is evenly divisible by the number of bands
# to use for LSH.
# Your function should rewrite each input file to the output directory with the same name, but
# only writing documents that are either (a) not candidate duplicates and/or (b) are randomly selected to be retained from the clustered buckets. For example, if the input paths are a/1.txt and
# a/2.txt, and the output directory is b/, your function should write the files b/1.txt and b/2.txt.
# Implement the adapter [run_minhash_deduplication] and make sure it passes uv run pytest -k
# test_minhash_deduplication.

import os
import hashlib
from typing import List

def normalize_text(text: str) -> str:
    import unicodedata
    import regex as re
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'\p{P}+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_ngrams(text: str, n: int) -> set:
    words = text.split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    return ngrams

def compute_minhash_signature(ngram_set: set, num_hashes: int) -> List[int]:
    if not ngram_set:
        return [float('inf')] * num_hashes
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for ngram in ngram_set:
            hash_value = int(hashlib.md5((str(i) + ngram).encode('utf-8')).hexdigest(), 16)
            if hash_value < min_hash:
                min_hash = hash_value
        signature.append(min_hash)
    return signature

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0
    
def minhash_deduplication(    
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    
    os.makedirs(output_directory, exist_ok=True)

    documents = []

    for file_idx, input_file in enumerate(input_files):
        with open(input_file, 'r', encoding='utf-8') as f:
            
            content = f.read()
            normalized_content = normalize_text(content)
            ngram_set = get_ngrams(normalized_content, ngrams)
            signature = compute_minhash_signature(ngram_set, num_hashes)
            documents.append({
                'file_idx': file_idx,
                'file_path': input_file,
                'original': content,
                'ngram_set': ngram_set,
                'signature': signature,
            })

    # LSH
    rows_per_bands = num_hashes // num_bands
    buckets = {}

    for doc_idx, doc in enumerate(documents):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_bands
            end = start + rows_per_bands
            band_signature = tuple(doc['signature'][start:end])
            bucket_key = (band_idx, band_signature)

            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(doc_idx)
    
    candidate_pairs = set()
    
    for bucket in buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    candidate_pairs.add((bucket[i], bucket[j]))

    to_remove = set()

    for i, j in candidate_pairs:
        if j in to_remove: continue
        sim = jaccard_similarity(documents[i]['ngram_set'], documents[j]['ngram_set'])
        if sim >= jaccard_threshold:
            to_remove.add(j)

    for doc_idx, doc in enumerate(documents):
        if doc_idx not in to_remove:
            output_path = os.path.join(output_directory, os.path.basename(doc['file_path']))
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(doc['original'])