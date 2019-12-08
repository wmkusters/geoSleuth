import json

loader = json.load(open("vars/adj_map.json"))
bin_adjacency_map = {int(k):v for k,v in loader.items()}