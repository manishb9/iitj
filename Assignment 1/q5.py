from collections import defaultdict, deque

MOD = 10**9 + 7

def countPathForNodes(V, E, S, edges):
    graph = defaultdict(list)
    incomingEdges = [0] * (V + 1)
    for u, v in edges:
        graph[u].append(v)
        incomingEdges[v] += 1

    queue = deque()
    for i in range(1, V + 1):
        if incomingEdges[i] == 0:
            queue.append(i)
    
    topOrder = []
    while queue:
        node = queue.popleft()
        topOrder.append(node)
        for neighbor in graph[node]:
            incomingEdges[neighbor] -= 1
            if incomingEdges[neighbor] == 0:
                queue.append(neighbor)

    dp = [0] * (V + 1)
    dp[S] = 1
    
    for u in topOrder:
        for v in graph[u]:
            dp[v] = (dp[v] + dp[u]) % MOD
    
    result = sum(dp) % MOD
    return result

if __name__ == "__main__":
    # First line: V, E, S
    V, E, S = map(int, input().split())
    
    edges = []
    for _ in range(E):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    result = countPathForNodes(V, E, S, edges)
    print(result)