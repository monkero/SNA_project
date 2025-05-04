def filter_file(input_file):
    output_file = input_file.replace(".vrt", "_filtered.txt")
    filter_list = {"työ", "työt", "töissä", "työssäkäyminen", "työpaikka"}
    matched_thread_ids = set()
    current_thread_id = None
    thread_contains_match = False

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.startswith("<text "):
                # Start of a new sentence block
                thread_contains_match = False
                current_thread_id = None

                # Extract thread_id from the line
                for part in line.split():
                    if part.startswith('thread_id="'):
                        current_thread_id = int(part.split('"')[1])
                        if current_thread_id in matched_thread_ids:
                            current_thread_id = None
                        break

            if current_thread_id:
                if not line.startswith("<") and line.strip():
                    # Check only first column (contains word)
                    # It's the only one we care about
                    word = line.split()[0].lower()
                    if word in filter_list:
                        thread_contains_match = True

                elif line.startswith("</text>"):
                    # End of sentence block
                    if thread_contains_match and current_thread_id:
                        matched_thread_ids.add(current_thread_id)

    with open(output_file, "w", encoding="utf-8") as fout:
        for thread_id in sorted(matched_thread_ids):
            fout.write(f"{thread_id}\n")


def main():
    input_file = "s24_2006.vrt"
    filter_file(input_file)


if __name__ == "__main__":
    main()
