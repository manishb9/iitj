from collections import defaultdict, deque

MOD = 10**9 + 7

def countPathForNodes(V, E, S, edges):
    graph = defaultdict(list)
    pathCount = [0] * (V + 1)
    incomingEdges = [0] * (V + 1)
    
    for u, v in edges:
        graph[u].append(v)
        incomingEdges[v] += 1
    
    queue = deque()
    queue.append(S)
    pathCount[S] = 1  
    
    while queue:
        node = queue.popleft()
        
        for neighbor in graph[node]:
            pathCount[neighbor] = (pathCount[neighbor] + pathCount[node]) % MOD
            incomingEdges[neighbor] -= 1
            
            if incomingEdges[neighbor] == 0:
                queue.append(neighbor)
    
    result = sum(pathCount) % MOD
    return result

if __name__ == "__main__":
    V, E, S = map(int, input().split())
    
    edges = []
    for i in range(E):
        u, v = map(int, input().split())
        edges.append((u, v))
    finalResult = countPathForNodes(V, E, S, edges)
    print(finalResult)
