from time import time
import re
from pathlib import Path
from matplotlib import pyplot as plt

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
                                      'total_count': 0,
                                      'title_count': 0}
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
                tmp = thread_title.split()
                for word in tmp:
                    if word in keywords:
                        keywords[word]['total_count'] += 1
                        keywords[word]['title_count'] += 1
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


def print_results(keywords, threads):
    print("RESULTS:")
    for key in keywords.keys():
        print(f"{key} : {keywords[key]['total_count']}")
        print(f"\t title count : {keywords[key]['title_count']}")
    print("\nTHREAD ID : Title")
    for id_ in threads.keys():
        print(f"{id_} : {threads[id_]}")


def keywords_to_histograms(results):
    keywords = list(results.keys())
    total_counts = [results[key]['total_count'] for key in keywords]
    title_counts = [results[key]['title_count'] for key in keywords]
    sorted_total_keys, sorted_total_counts = zip(*sorted(zip(keywords, total_counts), key=lambda x: x[1], reverse=True))
    sorted_title_keys, sorted_title_counts = zip(*sorted(zip(keywords, title_counts), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_total_keys, sorted_total_counts, color='skyblue')
    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.title('Keyword Frequency (Total)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_title_keys, sorted_title_counts, color='coral')
    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.title('Keyword Frequency (Titles)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
