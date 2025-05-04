import re
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx

# 1. Define violence keywords specific to Finnish workplace mental health
violence_keywords = {
    'väkivalta', 'uhkaus', 'pahoinpitely', 'kiusaaminen',
    'häirintä', 'painostus', 'uhka', 'aggressio',
    'hyökkäys', 'lyöminen', 'hakkaaminen', 'väkivaltainen'
}

# 2. Modified VRT parser to capture timestamps and violence mentions
def parse_vrt_with_violence(file_path):
    thread_data = defaultdict(lambda: {'violence_count': 0, 'timestamps': []})
    current_thread = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Start of thread
            if line.startswith('<text id="'):
                match = re.search(r'id="([^"]+)"', line)
                if match:
                    current_thread = match.group(1)
            
            # End of thread
            elif line.startswith('</text>'):
                current_thread = None
            
            # Process post content
            elif current_thread and line and not line.startswith('<'):
                parts = line.split('\t')
                if len(parts) >= 4:  # Assuming format: word\tlemma\t...\ttimestamp
                    word = parts[0].lower()
                    lemma = parts[1].lower()
                    timestamp = parts[3]  # Adjust index based on your VRT format
                    
                    # Check for violence keywords
                    if word in violence_keywords or lemma in violence_keywords:
                        thread_data[current_thread]['violence_count'] += 1
                        thread_data[current_thread]['timestamps'].append(timestamp)
    
    return thread_data

# 3. Enhanced reciprocity analysis with temporal patterns
def analyze_reciprocity(thread_data):
    reciprocity_results = {
        'fulfilled': 0,
        'unfulfilled': 0,
        'attack_pairs': []
    }
    
    # First pass: Simple even/odd count
    for tid, data in thread_data.items():
        if data['violence_count'] % 2 == 0 and data['violence_count'] > 0:
            reciprocity_results['fulfilled'] += 1
        elif data['violence_count'] > 0:
            reciprocity_results['unfulfilled'] += 1
    
    # Second pass: Temporal analysis of attack sequences
    for tid, data in thread_data.items():
        if len(data['timestamps']) >= 2:
            # Sort by timestamp
            try:
                sorted_times = sorted(
                    data['timestamps'],
                    key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')  # Adjust format
                )
                # Check for alternating pattern (attack -> counterattack)
                for i in range(len(sorted_times)-1):
                    time1 = datetime.strptime(sorted_times[i], '%Y-%m-%d %H:%M:%S')
                    time2 = datetime.strptime(sorted_times[i+1], '%Y-%m-%d %H:%M:%S')
                    hours_diff = (time2 - time1).total_seconds() / 3600
                    
                    # Considered reciprocal if within 72 hours (adjust as needed)
                    if hours_diff < 72:
                        reciprocity_results['attack_pairs'].append((tid, hours_diff))
            except ValueError as e:
                print(f"Timestamp parsing error in thread {tid}: {e}")
    
    return reciprocity_results

# 4. Visualization
def plot_reciprocity(results):
    plt.figure(figsize=(10, 6))
    
    # Bar plot for basic reciprocity
    plt.subplot(1, 2, 1)
    plt.bar(['Fulfilled', 'Not Fulfilled'], 
            [results['fulfilled'], results['unfulfilled']],
            color=['green', 'red'])
    plt.title('Reciprocity by Keyword Count')
    plt.ylabel('Number of Threads')
    
    # Temporal analysis plot if we have attack pairs
    if results['attack_pairs']:
        plt.subplot(1, 2, 2)
        time_diffs = [diff for _, diff in results['attack_pairs']]
        plt.hist(time_diffs, bins=20, color='purple')
        plt.title('Time Between Attacks')
        plt.xlabel('Hours')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('reciprocity_analysis.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and parse data
    thread_data = parse_vrt_with_violence("s24_2001_updated.vrt")
    
    # Analyze reciprocity
    results = analyze_reciprocity(thread_data)
    
    # Print summary statistics
    print(f"Total threads with violence mentions: {len(thread_data)}")
    print(f"Threads with fulfilled reciprocity: {results['fulfilled']}")
    print(f"Threads with unfulfilled reciprocity: {results['unfulfilled']}")
    print(f"Detected attack-response pairs: {len(results['attack_pairs'])}")
    
    # Generate visualizations
    plot_reciprocity(results)