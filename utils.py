import json
from itertools import combinations
from time import time
import re
from pathlib import Path
import os
import networkx as nx
from matplotlib import pyplot as plt

METRICS_FILE = "network_metrics.json"
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


def analyze_network(G, network_metrics, year):
    print("\nNETWORK ANALYSIS:")
    avg_deg = sum(G.degree(node) for node in G.nodes) / len(G.nodes)
    network_metrics['average degree'] = avg_deg
    print(f"Average degree: {avg_deg:.2f}")

    max_deg = max(G.degree(), key=lambda x: x[1])[1]
    network_metrics['maximum degree'] = max_deg
    print(f"Maximum degree: {max_deg}")

    global_cc = nx.transitivity(G)
    network_metrics['global clustering coefficient'] = global_cc
    print(f"Global clustering coefficient: {global_cc:.2f}")

    largest_comp = len(max(nx.connected_components(G), key=len))
    network_metrics['largest component size'] = largest_comp
    print(f"Largest component size: {largest_comp}")

    communities = 0
    for _ in nx.connected_components(G):
        communities += 1
    network_metrics['number communities'] = communities
    print(f"Number communities: {communities}")
    save_network_metrics(network_metrics, year)


def save_network_metrics(metrics, year):
    current_path = os.path.dirname(os.path.realpath(__file__))
    metrics_path = str(os.path.join(current_path, "results\\" + METRICS_FILE))
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as rfile:
            data = json.load(rfile)
    else:
        data = {}
    if year not in data:
        data[year] = metrics
        with open(metrics_path, "w", encoding="utf-8") as wfile:
            json.dump(data, wfile, indent=2)


def build_thread_network(keywords, threads, year):
    print("Creating graph...")
    G = nx.Graph()
    network_metrics = {}
    for thread_id, thread_title in threads.items():
        G.add_node(thread_id, label=thread_title)
    print(f"Added {G.number_of_nodes()} nodes")
    network_metrics['nodes'] = G.number_of_nodes()
    for keyword, data in keywords.items():
        thread_ids = data['thread_ids']
        for t1, t2 in combinations(thread_ids, 2):
            if G.has_edge(t1, t2):
                continue
            else:
                G.add_edge(t1, t2)
    print(f"Added {G.number_of_edges()} edges")
    network_metrics['edges'] = G.number_of_edges()
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = str(os.path.join(current_path, "results\\" + year))
    os.makedirs(dir_path, exist_ok=True)
    graph_path = str(os.path.join(dir_path, f"{year}_keywords_network.gexf"))
    if not os.path.exists(graph_path):
        nx.write_gexf(G, graph_path)
    return G, network_metrics


def show_network_graph(G):
    plt.figure(figsize=(15, 10))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos, with_labels=False, node_size=100, alpha=0.5)
    plt.show()


def print_results(keywords, threads):
    print("RESULTS:")
    for key in keywords.keys():
        print(f"{key} : {keywords[key]['total_count']}")
        print(f"\t title count : {keywords[key]['title_count']}")
    print("\nTHREAD ID : Title")
    for id_ in threads.keys():
        print(f"{id_} : {threads[id_]}")


def keywords_to_histograms(results, year):
    # Create directory for save data
    current_path = os.path.dirname(os.path.realpath(__file__))
    file_path = str(os.path.join(current_path, "results\\" + year))
    os.makedirs(file_path, exist_ok=True)
    fig_path = os.path.join(file_path, f"{year}_total_frequency.png")

    # Sort keywords and their counts
    keywords = list(results.keys())
    total_counts = [results[key]['total_count'] for key in keywords]
    title_counts = [results[key]['title_count'] for key in keywords]
    sorted_total_keys, sorted_total_counts = zip(*sorted(zip(keywords, total_counts), key=lambda x: x[1], reverse=True))
    sorted_title_keys, sorted_title_counts = zip(*sorted(zip(keywords, title_counts), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_total_keys, sorted_total_counts, color='skyblue')
    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.title('Keyword Frequency (Total) in ' + year)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save total figure
    if not os.path.exists(fig_path):
        plt.savefig(fig_path)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_title_keys, sorted_title_counts, color='coral')
    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.title('Keyword Frequency (Titles) in ' + year)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save title figure
    fig_path = os.path.join(file_path, f"{year}_title_frequency.png")
    if not os.path.exists(fig_path):
        plt.savefig(fig_path)
    plt.show()
    plt.close()
