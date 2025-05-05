from time import time
import re
from pathlib import Path

KEYWORDS_FILE = "keywords.txt"
REF_SIZE = 39260267748
REF_LINES = 514000000


def timed(func):
    start = time()
    result = func()
    return result, time() - start


def keywords_parser() -> dict[str, dict[str, set[int] | int]]:
    keywords = {}
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            keywords[line.strip()] = {'thread_ids': set(),
                                      'total_count': 0}
    return keywords


def keyword_matching(input_file, keywords) -> dict[str, dict[str, set[int] | int]] and dict[int, str]:
    file_size = Path(input_file).stat().st_size
    est_lines = int((file_size / REF_SIZE) * REF_LINES)
    thread_id = None
    thread_title = None
    threads = {}

    with open(input_file, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file):
            if line_num > 1 and line_num % 10_000_000 == 0:
                print(f"{(line_num / est_lines) * 100:.1f}%")
            if line.startswith("<text "):
                # Start of a new sentence block
                attributes = dict(re.findall(r'(\w+)="([^"]*)"', line))
                thread_id = int(attributes.get('thread_id'))
                thread_title = attributes.get('title')
            elif thread_id:
                fields = line.strip().split("\t")
                if len(fields) >= 3:
                    word = fields[0].lower()
                    lemma = fields[2].lower()

                    # Single-word match
                    if word in keywords or lemma in keywords:
                        keywords[word if word in keywords else lemma]['total_count'] += 1
                        keywords[word if word in keywords else lemma]['thread_ids'].add(thread_id)
                        threads[thread_id] = thread_title

    return keywords, threads
