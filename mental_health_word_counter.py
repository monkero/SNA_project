from collections import Counter

# Define keywords (normalize to lowercase and strip leading/trailing spaces)
keywords = {
    'mielenterveys', 'työhyvinvointi', 'työkyky', 'työssä jaksaminen',
    'työuupumus', 'stressinhallinta', 'palautuminen', 'työilmapiiri',
    'työyhteisö', 'työn imu', 'työmotivaatio', 'psyykkinen kuormitus',
    'työssä viihtyminen', 'työtyytyväisyys', 'työrauha', 'resilienssi',
    'työn merkityksellisyys', 'työn hallinta'
}

# Normalize keywords by splitting multiword expressions
single_word_keywords = set()
multi_word_keywords = []

for kw in keywords:
    if ' ' in kw:
        multi_word_keywords.append(tuple(kw.split()))
    else:
        single_word_keywords.add(kw)

counts = Counter()

# Buffers
current_text_block = []
matching_text_blocks = []
buffer = []
inside_text = False
thread_has_hit = False

with open("s24_2001_updated.vrt", "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()

        # Track thread start
        if stripped.startswith("<text"):
            inside_text = True
            current_text_block = [line]
            buffer = []
            thread_has_hit = False
            continue

        elif stripped.startswith("</text>"):
            current_text_block.append(line)
            if thread_has_hit:
                matching_text_blocks.extend(current_text_block)
            inside_text = False
            continue

        # Always buffer the current line in thread
        if inside_text:
            current_text_block.append(line)

        # Process token lines
        if inside_text and not line.startswith("<") and not line.startswith("#") and line.strip():
            fields = line.strip().split("\t")
            if len(fields) >= 3:
                word = fields[0].lower()
                lemma = fields[2].lower()
                buffer.append((word, lemma))

                # Single-word match
                if word in single_word_keywords or lemma in single_word_keywords:
                    counts[word if word in single_word_keywords else lemma] += 1
                    thread_has_hit = True

                # Multi-word match
                for phrase in multi_word_keywords:
                    if len(buffer) >= len(phrase):
                        window = buffer[-len(phrase):]
                        word_seq = tuple(w[0] for w in window)
                        lemma_seq = tuple(w[1] for w in window)
                        if word_seq == phrase or lemma_seq == phrase:
                            counts[" ".join(phrase)] += 1
                            thread_has_hit = True

# Print sorted counts
for term, count in counts.most_common():
    print(f"{term}: {count}")
