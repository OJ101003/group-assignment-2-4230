from mpi4py import MPI
import numpy as np

def read_edges_and_map_ids(filename):
    edges = []
    id_map = {}
    reverse_id_map = {}
    current_id = 0
    

    with open(filename, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.split())
            if node1 not in id_map:
                id_map[node1] = current_id
                reverse_id_map[current_id] = node1
                current_id += 1
            if node2 not in id_map:
                id_map[node2] = current_id
                reverse_id_map[current_id] = node2
                current_id += 1
            edges.append((id_map[node1], id_map[node2]))
    
    return edges, id_map, reverse_id_map

def create_adjacency_matrix(n, edges):
    inf = 999999
    A = np.full((n, n), inf, dtype=int)
    np.fill_diagonal(A, 0)
    for i, j in edges:
        A[i][j] = 1
        A[j][i] = 1
    return A

def calculate_closeness_centrality(n, shortest_paths):
    closeness_centrality = np.zeros(n)
    for i in range(n):
        sum_distances = np.sum(shortest_paths[i][shortest_paths[i] < 999999])  # Ignore 'inf' distances
        if sum_distances > 0:
            closeness_centrality[i] = (n - 1) / sum_distances
        else:
            closeness_centrality[i] = 0
    return closeness_centrality

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Read edges and map node IDs
edges, id_map, reverse_id_map = read_edges_and_map_ids('twitter_combined.txt')
n = len(id_map)  # Number of unique nodes

if size > n:
    if rank == 0:
        print(f"Error: Number of processes ({size}) exceeds number of nodes ({n}).")
    MPI.Finalize()
    exit()

# Start timing
start_time = MPI.Wtime()

if rank == 0:
    A = create_adjacency_matrix(n, edges)
else:
    A = np.empty((n, n), dtype=int)

rows_per_proc = [n // size + (1 if x < n % size else 0) for x in range(size)]
displs = [sum(rows_per_proc[:x]) * n for x in range(size)]
sendcounts = [rows_per_proc[x] * n for x in range(size)]

local_n = rows_per_proc[rank]
local_A = np.empty((local_n, n), dtype=int)
comm.Scatterv([A, sendcounts, displs, MPI.INT], local_A, root=0)

# print(f"Process {rank} local_A after Scatterv: \n{local_A}\n")

for k in range(n):
    row_k = np.empty(n, dtype=int)
    row_owner = np.where(k < np.cumsum([0] + rows_per_proc))[0][0] - 1

    if rank == row_owner:
        local_row_index = k - sum(rows_per_proc[:row_owner])
        row_k[:] = local_A[local_row_index, :]
    comm.Bcast(row_k, root=row_owner)

    for i in range(local_n):
        for j in range(n):
            local_A[i, j] = min(local_A[i, j], local_A[i, k] + row_k[j])

if rank == 0:
    result = np.empty((n, n), dtype=int)
else:
    result = None

comm.Gatherv(local_A, [result, sendcounts, displs, MPI.INT], root=0)

# End timing
end_time = MPI.Wtime()

if rank == 0:
    # Calculate closeness centrality
    closeness_centrality = calculate_closeness_centrality(n, result)
    
    # Print the total time taken
    print(f"Total time: {end_time - start_time} seconds")

    # Sort closeness centrality in descending order
    sorted_indices = np.argsort(closeness_centrality)[::-1]
    sorted_closeness_centrality = closeness_centrality[sorted_indices]

    with open("outputTW.txt", "w") as f:
        f.write(f"Top 5 nodes with highest closeness centrality:\n")
        for i in range(5):
            original_id = reverse_id_map[sorted_indices[i]]
            f.write(f"{original_id}\t{sorted_closeness_centrality[i]}\n")
        f.write(f"Average closeness centrality: {np.mean(closeness_centrality)}\n")
        f.write("List of all nodes and their closeness centrality:\n")
        for i, cc in enumerate(sorted_closeness_centrality):
            original_id = reverse_id_map[sorted_indices[i]]
            f.write(f"{original_id}\t{cc}\n")