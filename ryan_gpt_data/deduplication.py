import os
import hashlib
from typing import List

def run_exact_line_deduplication(input_files: List[str], output_dir: str) -> None:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    line_hash_count = {}
    
    # First pass: Count frequency of each line using hashes
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                line_hash_count[line_hash] = line_hash_count.get(line_hash, 0) + 1
    
    # Second pass: Rewrite each file with unique lines only
    for input_file in input_files:
        output_file_path = os.path.join(output_dir, os.path.basename(input_file))
        with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.rstrip('\n')
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                if line_hash_count[line_hash] == 1:
                    f_out.write(line + '\n')
                    
