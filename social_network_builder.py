import re
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

# Define your keyword sets
keywords = {
    'mielenterveys', 'työhyvinvointi', 'työkyky', 'työssä jaksaminen',
    'työuupumus', 'stressinhallinta', 'palautuminen', 'työilmapiiri',
    'työyhteisö', 'työn imu', 'työmotivaatio', 'psyykkinen kuormitus',
    'työssä viihtyminen', 'työtyytyväisyys', 'työrauha', 'resilienssi',
    'työn merkityksellisyys', 'työn hallinta'
}

single_keywords = {kw for kw in keywords if ' ' not in kw}
multi_keywords = [tuple(kw.split()) for kw in keywords if ' ' in kw]

# STEP 1: Parse .vrt and build thread -> keywords mapping
thread_keywords = defaultdict(set)
thread_timestamps = defaultdict(list)  # To store timestamps for reciprocity analysis
buffer = []
thread_id = None

with open("s24_2001_updated.vrt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line.startswith("<text"):
            match = re.search(r'id="([^"]+)"', line)
            if match:
                thread_id = match.group(1)
                buffer = []
            else:
                thread_id = None

        elif line.startswith("</text>"):
            if thread_id:
                # Check multi-word keyword matches
                for i in range(len(buffer)):
                    for phrase in multi_keywords:
                        if i + len(phrase) <= len(buffer):
                            word_seq = tuple(w for w, _ in buffer[i:i+len(phrase)])
                            lemma_seq = tuple(l for _, l in buffer[i:i+len(phrase)])
                            if word_seq == phrase or lemma_seq == phrase:
                                thread_keywords[thread_id].add(" ".join(phrase))
            buffer = []
            thread_id = None

        elif line.startswith("<") or not line or thread_id is None:
            continue

        else:
            parts = line.split("\t")
            if len(parts) >= 3:
                word = parts[0].lower()
                lemma = parts[2].lower()
                buffer.append((word, lemma))
                if word in single_keywords or lemma in single_keywords:
                    keyword = word if word in single_keywords else lemma
                    thread_keywords[thread_id].add(keyword)
                    # Store timestamp if available (assuming it's in parts[3])
                    if len(parts) > 3:
                        thread_timestamps[thread_id].append(parts[3])

# STEP 2: Threshold-based network analysis
def analyze_with_thresholds(thread_keywords, k_values):
    metrics = {
        'k': [],
        'nodes': [],
        'edges': [],
        'avg_degree': [],
        'giant_component': [],
        'avg_path_length': [],
        'diameter': [],
        'clustering': [],
        'shared_keyword_counts': []  # New: Track actual shared keyword counts
    }
    
    for k in k_values:
        G = nx.Graph()
        shared_counts = []  # Track how many keyword pairs share ≥k keywords
        
        # Add all threads as nodes
        for tid in thread_keywords:
            G.add_node(tid)
        
        # Compare all thread pairs
        thread_ids = list(thread_keywords.keys())
        for i in range(len(thread_ids)):
            for j in range(i + 1, len(thread_ids)):
                tid1, tid2 = thread_ids[i], thread_ids[j]
                shared = thread_keywords[tid1] & thread_keywords[tid2]
                shared_counts.append(len(shared))
                
                if len(shared) >= k:
                    G.add_edge(tid1, tid2, weight=len(shared))
        
        # Store metrics
        metrics['k'].append(k)
        metrics['nodes'].append(G.number_of_nodes())
        metrics['edges'].append(G.number_of_edges())
        metrics['shared_keyword_counts'].append(shared_counts)  # Store distribution
        
        degrees = dict(G.degree())
        metrics['avg_degree'].append(sum(degrees.values()) / max(1, len(degrees)))
        metrics['clustering'].append(nx.average_clustering(G) if G.number_of_edges() > 0 else 0)
        
        # Handle connected components
        if nx.is_connected(G):
            metrics['giant_component'].append(G.number_of_nodes())
            metrics['avg_path_length'].append(nx.average_shortest_path_length(G))
            metrics['diameter'].append(nx.diameter(G))
        else:
            largest_cc = max(nx.connected_components(G), key=len) if G.number_of_nodes() > 0 else set()
            metrics['giant_component'].append(len(largest_cc))
            if len(largest_cc) > 1:
                subgraph = G.subgraph(largest_cc)
                metrics['avg_path_length'].append(nx.average_shortest_path_length(subgraph))
                metrics['diameter'].append(nx.diameter(subgraph))
            else:
                metrics['avg_path_length'].append(0)
                metrics['diameter'].append(0)
    
    return metrics

# Analyze with different thresholds
k_values = [1, 2, 3, 5]
metrics = analyze_with_thresholds(thread_keywords, k_values)

# Print debug information
print("\nShared keyword distribution across all thread pairs:")
for k, counts in zip(metrics['k'], metrics['shared_keyword_counts']):
    print(f"k={k}: Max shared {max(counts) if counts else 0}, "
          f"Pairs with ≥{k} shared: {sum(1 for c in counts if c >= k)}")

# Plot threshold evolution (only metrics with non-zero values)
plt.figure(figsize=(12, 8))
metrics_to_plot = [m for m in ['avg_degree', 'giant_component', 'avg_path_length', 'diameter', 'clustering'] 
                  if max(metrics[m]) > 0]

for i, metric in enumerate(metrics_to_plot):
    plt.subplot(2, 3, i+1)
    plt.plot(metrics['k'], metrics[metric], 'o-')
    plt.title(metric.replace('_', ' ').title())
    plt.xlabel('Threshold (k)')
    plt.ylim(bottom=0)  # Start y-axis at 0

plt.tight_layout()
plt.savefig("threshold_evolution.png")
plt.show()

# STEP 3: Reciprocity analysis
def analyze_reciprocity(thread_keywords, thread_timestamps):
    fulfilled = 0
    not_fulfilled = 0
    
    for tid, kws in thread_keywords.items():
        # Count violence keywords (modify this filter as needed)
        violence_kws = [kw for kw in kws if 'violence' in kw.lower()]  # Adjust keyword filter
        if len(violence_kws) % 2 == 0:
            fulfilled += 1
        else:
            not_fulfilled += 1
    
    # Plot reciprocity
    plt.figure(figsize=(8, 6))
    plt.bar(['Reciprocity fulfilled', 'Not fulfilled'], [fulfilled, not_fulfilled])
    plt.title("Reciprocity Analysis")
    plt.ylabel("Number of Threads")
    plt.savefig("reciprocity_analysis.png")
    plt.show()

analyze_reciprocity(thread_keywords, thread_timestamps)

# STEP 4: Original network statistics (with k=1)
G = nx.Graph()
for tid in thread_keywords:
    G.add_node(tid)

thread_ids = list(thread_keywords.keys())
for i in range(len(thread_ids)):
    for j in range(i + 1, len(thread_ids)):
        tid1, tid2 = thread_ids[i], thread_ids[j]
        shared = thread_keywords[tid1] & thread_keywords[tid2]
        if shared:
            G.add_edge(tid1, tid2)

print("\nNetwork Statistics (k=1):")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

degrees = dict(G.degree())
print("Max degree:", max(degrees.values()))
print("Average degree:", sum(degrees.values()) / len(degrees))

print("Clustering coefficient:", nx.average_clustering(G))

if nx.is_connected(G):
    print("Diameter:", nx.diameter(G))
    print("Average path length:", nx.average_shortest_path_length(G))
else:
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    print("Diameter (giant component):", nx.diameter(subgraph))
    print("Average path length (giant component):", nx.average_shortest_path_length(subgraph))

# Communities
communities = list(greedy_modularity_communities(G))
print("\nCommunity Analysis:")
print("Number of communities:", len(communities))
print("Size of largest community:", max(len(c) for c in communities))
print("Modularity:", nx.algorithms.community.modularity(G, communities))

# Degree distribution
plt.figure(figsize=(8, 6))
plt.hist(degrees.values(), bins=30, color="steelblue", edgecolor="black")
plt.title("Degree Distribution of Thread Network")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.tight_layout()
plt.savefig("degree_distribution.png")
plt.show()