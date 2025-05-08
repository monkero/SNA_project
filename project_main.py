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
    keywords, threads = result
    if print_interim:
        print_results(keywords, threads)
    keywords_to_histograms(keywords, year)
    G, network_metrics = build_thread_network(keywords, threads, year)
    if show_graph:
        show_network_graph(G)
    network_metrics = analyze_network(G, network_metrics)
    degrees = plot_degree_distribution(G, year)
    powerlaw_fit(network_metrics, degrees, year)
    analyze_network_with_threshold(keywords, threads, year)
    analyze_reciprocity(threads, year)
    print("\nDone.")


if __name__ == "__main__":
    main()
