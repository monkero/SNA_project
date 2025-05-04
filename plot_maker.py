import matplotlib.pyplot as plt
from collections import Counter

# Define topic keywords
keywords = {
    'mielenterveys', 'työhyvinvointi', 'työkyky', 'työssä jaksaminen',
    'työuupumus', 'stressinhallinta', 'palautuminen', 'työilmapiiri',
    'työyhteisö', 'työn imu', 'työmotivaatio', 'psyykkinen kuormitus',
    'työssä viihtyminen', 'työtyytyväisyys', 'työrauha', 'resilienssi',
    'työn merkityksellisyys', 'työn hallinta'
}

# Normalize keywords
keywords = {kw.lower() for kw in keywords}
single_word_keywords = {kw for kw in keywords if " " not in kw}
multi_word_keywords = [tuple(kw.split()) for kw in keywords if " " in kw]

# Counters
fulltext_counts = Counter()
title_counts = Counter()

# Read filtered file
with open("s24_2001_updated.vrt", "r", encoding="utf-8") as f:
    buffer = []
    in_text = False
    current_title = ""

    for line in f:
        line = line.strip()

        if line.startswith("<text"):
            in_text = True
            buffer = []
            # Extract thread title if available
            if 'title="' in line:
                start = line.index('title="') + 7
                end = line.index('"', start)
                current_title = line[start:end].lower()
            else:
                current_title = ""
            continue
        elif line.startswith("</text>"):
            in_text = False
            buffer = []
            continue
        elif not in_text or line.startswith("<") or line.startswith("#") or not line.strip():
            continue

        fields = line.split("\t")
        if len(fields) < 3:
            continue
        word = fields[0].lower()
        lemma = fields[2].lower()
        buffer.append((word, lemma))

        # Single-word match
        for form in (word, lemma):
            if form in single_word_keywords:
                fulltext_counts[form] += 1

        # Multi-word match
        for phrase in multi_word_keywords:
            if len(buffer) >= len(phrase):
                window = buffer[-len(phrase):]
                word_seq = tuple(w[0] for w in window)
                lemma_seq = tuple(w[1] for w in window)
                if word_seq == phrase or lemma_seq == phrase:
                    fulltext_counts[" ".join(phrase)] += 1

    # Check title keywords
    for kw in keywords:
        if kw in current_title:
            title_counts[kw] += 1

# --- Plot 1: All Posts ---
plt.figure(figsize=(12, 6))
plt.bar(fulltext_counts.keys(), fulltext_counts.values(), color='skyblue')
plt.xticks(rotation=90)
plt.title("Keyword Frequency in All Posts")
plt.xlabel("Keyword")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("keyword_frequency_posts.png")
plt.show()

# --- Plot 2: Thread Titles ---
plt.figure(figsize=(12, 6))
plt.bar(title_counts.keys(), title_counts.values(), color='lightcoral')
plt.xticks(rotation=90)
plt.title("Keyword Frequency in Thread Titles")
plt.xlabel("Keyword")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("keyword_frequency_titles.png")
plt.show()
