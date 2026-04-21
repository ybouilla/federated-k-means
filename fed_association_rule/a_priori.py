from itertools import combinations
from collections import defaultdict

# -----------------------------
# Local Apriori (on each client)
# -----------------------------
def get_frequent_itemsets(transactions, min_support):
    # TODO: to get  min support thresh value, one can use
    #  use median or 75th percentile of aggregated supports
    itemset_counts = defaultdict(int)
    total_transactions = len(transactions)

    # Generate all itemsets (simplified: up to size 2 for clarity)
    for transaction in transactions:
        transaction = set(transaction)

        # size-1 itemsets
        for item in transaction:
            itemset_counts[(item,)] += 1

        # size-2 itemsets
        for pair in combinations(transaction, 2):
            itemset_counts[tuple(sorted(pair))] += 1

    # Compute frequent itemsets
    frequent_itemsets = {}
    for itemset, count in itemset_counts.items():
        support = count / total_transactions
        if support >= min_support:
            frequent_itemsets[itemset] = count

    return frequent_itemsets


# -----------------------------
# Federated Server Aggregation
# -----------------------------
def federated_aggregate(client_outputs):
    global_counts = defaultdict(int)

    # sum counts from all clients
    for client_data in client_outputs:
        for itemset, count in client_data.items():
            global_counts[itemset] += count

    return global_counts


def compute_global_support(global_counts, total_transactions_all_clients):
    return {
        itemset: count / total_transactions_all_clients
        for itemset, count in global_counts.items()
    }


# -----------------------------
# Example Simulation
# -----------------------------

# Each client has its own private dataset
client_1 = [
    ["milk", "bread", "butter"],
    ["milk", "bread"],
    ["bread", "butter"]
]

client_2 = [
    ["milk", "bread"],
    ["milk", "diaper"],
    ["milk", "bread", "diaper"]
]

client_3 = [
    ["bread", "butter"],
    ["milk", "bread"],
    ["bread", "diaper"]
]

clients = [client_1, client_2, client_3]

min_support = 0.5

# Step 1: local computation
client_outputs = []
total_transactions_all_clients = 0

for client_data in clients:
    client_outputs.append(get_frequent_itemsets(client_data, min_support))
    total_transactions_all_clients += len(client_data)

# Step 2: server aggregation
global_counts = federated_aggregate(client_outputs)

# Step 3: compute global support
global_frequent_itemsets = compute_global_support(
    global_counts,
    total_transactions_all_clients
)

# Step 4: filter final frequent itemsets
final_result = {
    k: v for k, v in global_frequent_itemsets.items()
    if v >= min_support
}

print("🌍 Global Frequent Itemsets:")
for itemset, support in final_result.items():
    print(f"{itemset}: {support:.2f}")