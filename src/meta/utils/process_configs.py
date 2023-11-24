import sys
import pytoml

def process_query_cfg(cfg):
    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d.get('query-runner')
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    return query_path, query_start

def process_param_tuner_cfg(cfg):
    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    tuner_cfg = cfg_d.get('params-tuner')
    if tuner_cfg is None:
        print("params_tuner table needed in {}".format(cfg))
        sys.exit(1)

    logging_level = tuner_cfg.get('logging-level', 0)
    split_ratio = tuner_cfg.get('split-ratio', 0.8)
    cv = tuner_cfg.get('cv', 5)

    return logging_level, split_ratio, cv