import re

# Define work-related words (surface forms and lemmas)
work_keywords = {'työ', 'työt', 'töissä', 'työssäkäyminen'}

# Flags and buffers
inside_text_block = False
current_block = []
matched_blocks = []

with open("s24_2001.vrt", "r", encoding="utf-8") as infile, open("s24_2001_updated.vrt", "w", encoding="utf-8") as outfile:
    for line in infile:
        if line.startswith("<text "):
            inside_text_block = True
            current_block = [line]
            matched = False
        elif line.startswith("</text>") and inside_text_block:
            current_block.append(line)
            inside_text_block = False
            if matched:
                outfile.writelines(current_block)
        elif inside_text_block:
            current_block.append(line)
            # Skip metadata or empty lines
            if line.startswith("<") or line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) >= 3:
                word_form = fields[0].lower()
                lemma = fields[2].lower()
                if word_form in work_keywords or lemma in work_keywords:
                    matched = True
