from utils import keywords_parser, keyword_matching, timed, print_results, keywords_to_histograms, build_thread_network, show_network_graph



def main():
    # Args
    print_interim = False
    show_graph = True
    year = "2001"
    input_file = f"s24_{year}.vrt"

    keywords = keywords_parser()
    print(f"Parsing {input_file} please wait...")
    result, time = timed(lambda: keyword_matching(input_file, keywords))
    print(f"Parsing {input_file} took {time:.1f}s\n")
    counts, threads = result
    if print_interim:
        print_results(counts, threads)
    keywords_to_histograms(counts, year)
    G = build_thread_network(counts, threads)
    if show_graph:
        show_network_graph(G)
    print("Done.\n")


if __name__ == "__main__":
    main()
