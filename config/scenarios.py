from copy import deepcopy


SCENARIOS = {
    "S1": {"chunk_size": 300, "overlap": 50, "top_k": 3, "desc": "Chunk kecil, k kecil"},
    "S2": {"chunk_size": 300, "overlap": 50, "top_k": 5, "desc": "Chunk kecil, k sedang"},
    "S3": {"chunk_size": 300, "overlap": 50, "top_k": 7, "desc": "Chunk kecil, k besar"},
    "S4": {"chunk_size": 500, "overlap": 50, "top_k": 3, "desc": "Chunk sedang, k kecil"},
    "S5": {"chunk_size": 500, "overlap": 50, "top_k": 5, "desc": "Chunk sedang, k sedang"},
    "S6": {"chunk_size": 500, "overlap": 50, "top_k": 7, "desc": "Chunk sedang, k besar"},
    "S7": {"chunk_size": 700, "overlap": 50, "top_k": 3, "desc": "Chunk besar, k kecil"},
    "S8": {"chunk_size": 700, "overlap": 50, "top_k": 5, "desc": "Chunk besar, k sedang"},
    "S9": {"chunk_size": 700, "overlap": 50, "top_k": 7, "desc": "Chunk besar, k besar"},
}


def get_scenario(scenario_id):
    config = SCENARIOS.get(scenario_id.upper())
    return deepcopy(config) if config else None


def get_all_scenarios():
    return deepcopy(SCENARIOS)