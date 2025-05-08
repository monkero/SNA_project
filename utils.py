import json
from itertools import combinations
from time import time
import re
from pathlib import Path
import os
import networkx as nx
import networkx.exception
from matplotlib import pyplot as plt
import community as cl
from collections import defaultdict, Counter
import powerlaw

METRICS_FILE = "network_metrics.json"
KEYWORDS_FILE = "keywords.txt"
REF_SIZE = 39260267748
REF_LINES = 514000000


def timed(func):
    start = time()
    result = func()
    return result, time() - start


def keywords_parser():
    keywords = {}
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            keywords[line.strip()] = {'thread_ids': set(),
                                      'total_count': 0,
                                      'title_count': 0}
    return keywords


def keyword_matching(input_file, keywords):
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

                    # Keyword matching
                    if word in keywords:
                        keywords[word]['total_count'] += 1
                        keywords[word]['thread_ids'].add(thread_id)
                        if thread_id not in threads:
                            threads[thread_id] = {'title': thread_title}
                        if word not in threads[thread_id]:
                            threads[thread_id][word] = 1
                        else:
                            threads[thread_id][word] += 1
                    if lemma in keywords:
                        keywords[lemma]['total_count'] += 1
                        keywords[lemma]['thread_ids'].add(thread_id)
                        if thread_id not in threads:
                            threads[thread_id] = {'title': thread_title}
                        if lemma not in threads[thread_id]:
                            threads[thread_id][lemma] = 1
                        else:
                            threads[thread_id][lemma] += 1

    return keywords, threads


def plot_degree_distribution(G, year):
    current_path = os.path.dirname(os.path.realpath(__file__))
    file_path = str(os.path.join(current_path, "results\\" + year))
    degrees = [d for _, d in G.degree()]

    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, color='yellowgreen', edgecolor='black')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    fig_path = os.path.join(file_path, f"{year}_degrees_distribution.png")
    if not os.path.exists(fig_path):
        plt.savefig(fig_path)
    plt.show()
    return degrees


def powerlaw_fit(network_metrics, degrees, year):
    fit = powerlaw.Fit(degrees, discrete=True)
    print("\nPower-law fit results for the degree distribution:")
    R_exp, P_exp = fit.distribution_compare('power_law', 'lognormal')
    print(f"  Estimated xmin: {fit.xmin}")
    print(f"  (R): {R_exp:.2f}")
    print(f"  (P): {P_exp:.4f}")
    print("  Positive R with low P (< 0.05) -> the degree distribution most likely follows power-law ")
    network_metrics['follows powerlaw'] = True if (R_exp > 0 and P_exp < 0.05) else False
    ax = fit.plot_ccdf(label='Empirical')
    fit.power_law.plot_ccdf(ax=ax, linestyle='--', label='Power law')
    fit.lognormal.plot_ccdf(ax=ax, linestyle=':', label='Lognormal')
    plt.legend()
    plt.title(f"Degree CCDF (year {year})")
    plt.show()
    save_network_metrics(network_metrics, year)


def analyze_network(G, network_metrics):
    """
    LCC = Largest connected component
    metrics = {
        'nodes',
        'edges',
        'maximum degree',
        "average degree",
        'global clustering coefficient',
        'diameter',
        'LCC diameter',
        'average path length',
        'community sizes',
        'number of communities',
        'modularity',
        'LCC size',
        'average degree centrality'
        'follows powerlaw'
    }
    """
    print("\nNETWORK ANALYSIS:")
    max_deg = max(G.degree(), key=lambda x: x[1])[1]
    network_metrics['maximum degree'] = max_deg
    print(f"Maximum degree: {max_deg}")

    avg_deg = sum(G.degree(node) for node in G.nodes) / len(G.nodes)
    network_metrics['average degree'] = avg_deg
    print(f"Average degree: {avg_deg:.3f}")

    global_cc = nx.transitivity(G)
    network_metrics['global clustering coefficient'] = global_cc
    print(f"Global clustering coefficient: {global_cc:.3f}")

    try:
        diameter = nx.diameter(G)
    except networkx.exception.NetworkXError:
        diameter = "inf"
    network_metrics['diameter'] = diameter
    print(f"Diameter of the graph: {diameter}")

    lcc = max(nx.connected_components(G), key=len)
    lcc_graph = G.subgraph(lcc).copy()
    lcc_diameter = nx.diameter(lcc_graph)
    lcc_size = len(lcc)
    network_metrics['LCC diameter'] = lcc_diameter
    print(f"Largest connected component diameter: {lcc_diameter}")

    if nx.is_connected(G) and len(G.nodes) > 1:
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        if lcc_size > 1:
            avg_path_length = nx.average_shortest_path_length(lcc_graph)
        else:
            avg_path_length = 0

    network_metrics['average path length'] = avg_path_length
    print(f"Average path length: {avg_path_length}")

    partition = cl.community_louvain.best_partition(G)
    communities = {}
    for node, id_ in partition.items():
        communities.setdefault(id_, []).append(node)
    modularity = cl.modularity(partition, G)
    communities = list(communities.values())
    communities_sizes = sorted([len(c) for c in communities], reverse=True)

    network_metrics['community sizes'] = communities_sizes
    print(f"Community sizes: {communities_sizes}")

    network_metrics['number communities'] = len(communities_sizes)
    print(f"Number communities: {len(communities_sizes)}")

    network_metrics['modularity'] = modularity
    print(f"Modularity (quality): {modularity:.3f}")

    network_metrics['LCC size'] = lcc_size
    print(f"Largest connected component size: {lcc_size}")

    deg_cent = nx.degree_centrality(G)
    avg_deg_cent = sum(deg_cent.values()) / len(deg_cent)
    network_metrics['average degree centrality'] = avg_deg_cent
    print(f"Average degree centrality: {avg_deg_cent:.3f}")

    return network_metrics


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
    for thread in threads.keys():
        G.add_node(thread, label=threads[thread]['title'])
    print(f"Added {G.number_of_nodes()} nodes")
    network_metrics['nodes'] = G.number_of_nodes()
    for keyword, data in keywords.items():
        thread_ids = data['thread_ids']
        for t1, t2 in combinations(thread_ids, 2):
            if G.has_edge(t1, t2):
                G[t1][t2]['weight'] += 1
            else:
                G.add_edge(t1, t2, weight=1)

    for node in G.nodes():
        node_keywords = [kw for kw, info in keywords.items() if node in info['thread_ids']]
        G.nodes[node]['keywords'] = ";".join(node_keywords)

    partition = cl.community_louvain.best_partition(G, weight='weight')
    communities = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    community_label = {}
    for cid, nodes in communities.items():
        all_kw = []
        for n in nodes:
            kw_list = G.nodes[n]['keywords'].split(';')
            all_kw.extend(kw_list)
        most_common, _ = Counter(all_kw).most_common(1)[0]
        community_label[cid] = most_common

    for node, cid in partition.items():
        G.nodes[node]['community_label'] = community_label[cid]

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


def analyze_network_with_threshold(keywords, threads, year):
    k_values = [2, 3, 5, 10]
    metrics = {
        'k': k_values,
        'nodes': [],
        'edges': [],
        'maximum degree': [],
        "average degree": [],
        'global clustering coefficient': [],
        'LCC diameter': [],
        'average path length': [],
        'community sizes': [],
        'number communities': [],
        'modularity': [],
        'LCC size': [],
        'average degree centrality': []
    }

    for k in k_values:
        G = nx.Graph()
        G.add_nodes_from(threads.keys())
        for keyword, data in keywords.items():
            thread_ids = data['thread_ids']
            for t1, t2 in combinations(thread_ids, 2):
                if G.has_edge(t1, t2):
                    continue
                if threads[t1][keyword] > k - 1 and threads[t2][keyword] > k - 1:
                    G.add_edge(t1, t2)
        metrics['nodes'].append(G.number_of_nodes())
        metrics['edges'].append(G.number_of_edges())
        calculate_metrics(metrics, G)
    display_metrics(metrics, year)


def display_metrics(metrics, year):
    current_path = os.path.dirname(os.path.realpath(__file__))
    file_path = str(os.path.join(current_path, "results\\" + year))
    plt.figure(figsize=(12, 8))
    for key in ['nodes', 'edges', 'maximum degree',
                'LCC size', 'number communities']:
        plt.plot(metrics['k'], metrics[key], marker='o', label=key)

    plt.xlabel('Threshold k')
    plt.ylabel('Value')
    plt.title('Core Network Metrics vs Threshold k (figure 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(file_path, f"{year}_threshold_k_figure_(1).png")
    if not os.path.exists(fig_path):
        plt.savefig(fig_path)
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 8))
    for key in ['average degree centrality', 'modularity', 'average path length', 'LCC diameter',
                'average degree', 'global clustering coefficient']:
        plt.plot(metrics['k'], metrics[key], marker='s', label=key)

    plt.xlabel('Threshold k')
    plt.ylabel('Value')
    plt.title('Core Network Metrics vs Threshold k (figure 2)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(file_path, f"{year}_threshold_k_figure_(2).png")
    if not os.path.exists(fig_path):
        plt.savefig(fig_path)
    plt.show()
    plt.close()


def calculate_metrics(metrics, G):
    max_deg = max(G.degree(), key=lambda x: x[1])[1]
    avg_deg = sum(G.degree(node) for node in G.nodes) / len(G.nodes)
    gcc = nx.transitivity(G)

    components = list(nx.connected_components(G))
    giant_comp = max(components, key=len)
    giant_size = len(giant_comp)

    if nx.is_connected(G) and len(G.nodes) > 1:
        avg_path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        # Compute for the giant component if exists
        if giant_size > 1:
            sub = G.subgraph(giant_comp)
            avg_path_length = nx.average_shortest_path_length(sub)
            diameter = nx.diameter(sub)
        else:
            avg_path_length = 0
            diameter = 0

    deg_cent = nx.degree_centrality(G)
    avg_deg_cent = sum(deg_cent.values()) / len(deg_cent)

    partition = cl.community_louvain.best_partition(G)
    communities = {}
    for node, id_ in partition.items():
        communities.setdefault(id_, []).append(node)
    try:
        modularity = cl.modularity(partition, G)
    except ValueError:
        modularity = 0
    communities = list(communities.values())
    communities_sizes = [len(c) for c in communities]

    metrics['maximum degree'].append(max_deg)
    metrics['average degree'].append(avg_deg)
    metrics['global clustering coefficient'].append(gcc)
    metrics['LCC diameter'].append(diameter)
    metrics['average path length'].append(avg_path_length)
    metrics['community sizes'].append(communities_sizes)
    metrics['number communities'].append(len(communities_sizes))
    metrics['modularity'].append(modularity)
    metrics['LCC size'].append(giant_size)
    metrics['average degree centrality'].append(avg_deg_cent)


def analyze_reciprocity(threads, year):
    current_path = os.path.dirname(os.path.realpath(__file__))
    file_path = str(os.path.join(current_path, "results\\" + year))
    even = 0
    all_threads = len(threads.keys())
    for tid in threads.keys():
        count = 0
        for kw in threads[tid].keys():
            if not kw == 'title':
                count += threads[tid][kw]
        if (count & 1) == 0:
            even += 1
    odd = all_threads - even

    categories = ['Even Amount of Threads', 'Odd Amount of Threads']
    counts = [even, odd]
    colors = ['#66b3ff', '#ff9999']

    plt.bar(categories, counts, color=colors)
    plt.xlabel('Thread Type')
    plt.ylabel('Number of Threads')
    plt.title('Distribution of Even/Odd Amount of Keywords in Threads')

    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center')

    fig_path = os.path.join(file_path, f"{year}_even_odd_figure.png")
    if not os.path.exists(fig_path):
        plt.savefig(fig_path)
    plt.show()

