
def train_test_split(cfg, query_path, query_start, split_ratio):
    # Fit and score the estimator
    with open(query_path) as query_file:
        queries = [line.strip() for line in query_file]
        split_index = int(len(queries) * split_ratio)
        queries_train = queries[:split_index]
        queries_test = queries[split_index:]
    return queries_train, queries_test, query_start, query_start + split_index
