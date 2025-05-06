from utils import *


def main():
    # Args
    print_interim = False
    show_graph = False
    year = "2001"
    input_file = f"s24_{year}.vrt"

    keywords = keywords_parser()
    print(f"Parsing {input_file} please wait...")
    result, time_ = timed(lambda: keyword_matching(input_file, keywords))
    print(f"Parsing {input_file} took {time_:.1f}s\n")
    counts, threads = result
    if print_interim:
        print_results(counts, threads)
    keywords_to_histograms(counts, year)
    G, network_metrics = build_thread_network(counts, threads, year)
    if show_graph:
        show_network_graph(G)
    analyze_network(G, network_metrics, year)
    print("\nDone.")


if __name__ == "__main__":
    main()
