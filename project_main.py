from utils import keywords_parser, keyword_matching, timed, print_results, keywords_to_histograms


def main():
    # Args
    print_interim = False
    input_file = "s24_2001.vrt"

    keywords = keywords_parser()
    print(f"Parsing {input_file} please wait...")
    result, time = timed(lambda: keyword_matching(input_file, keywords))
    print(f"Parsing {input_file} took {time:.1f}s\n")
    counts, threads = result
    if print_interim:
        print_results(counts, threads)
    keywords_to_histograms(counts)


if __name__ == "__main__":
    main()
