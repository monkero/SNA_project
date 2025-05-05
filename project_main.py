from utils import keywords_parser, keyword_matching, timed


def main():
    input_file = "s24_2001.vrt"
    keywords = keywords_parser()
    print(f"Parsing {input_file} please wait...")
    result, time = timed(lambda: keyword_matching(input_file, keywords))
    keywords, threads = result
    print(f"Parsing {input_file} took {time:.1f}s\n")
    print("RESULTS:")
    for key in keywords.keys():
        print(f"{key} : {keywords[key]['total_count']}")
    print("\nTHREAD ID : Title")
    for id in threads.keys():
        print(f"{id} : {threads[id]}")


if __name__ == "__main__":
    main()
