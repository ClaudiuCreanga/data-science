from typing import List
from collections import defaultdict
from heapq import heappush, heappop


def allPathsSourceTarget(graph: List[List[int]]) -> List[List[int]]:
    # dfs while loop
    # when looking for paths with dfs, it is good to keep path information in the stack for each member
    target = len(graph) - 1
    stack = [(0, [])]
    result = []
    while stack:
        vertex, path = stack.pop()
        path.append(vertex)
        if vertex == target:
            result.append(path)
        elif graph[vertex]:
            for edge in graph[vertex]:
                stack.append((edge, list(path)))

    return result

def allPathsSourceTarget2(graph: List[List[int]]) -> List[List[int]]:
    # dfs paths with recursion
    # when you update a result variable you don't have to return the dfs method in recursion
    target = len(graph) - 1
    result = []
    def dfs(item):
        vertex, path = item
        path.append(vertex)
        if vertex == target:
            result.append(path)
        elif graph[vertex]:
                for edge in graph[vertex]:
                    dfs((edge, list(path)))
    dfs((0, []))
    return result

#print(allPathsSourceTarget2([[1,2], [3], [3], []]))


def networkDelayTime2(times: List[List[int]], N: int, K: int) -> int:
    g = defaultdict(list)
    for item in times:
        source, target, time = item
        g[source].append([target, time])
    visited = {K: 0}
    queue = [(K, 0)]
    while queue:
        node, time = queue.pop(0)
        if node not in visited:
            visited[node] = time
        else:
            for v, t in g[node]:
                queue.append((v, time + t))
    if len(visited) == N:
        return max(visited.values())
    return -1

def networkDelayTime(times: List[List[int]], N: int, K: int) -> int:
    # heap with dijkstra priority queue
    # bfs, because dijkstra is a bfs with priority queue
    g = defaultdict(list)
    for source, target, time in times:
        g[source].append((target, time))
    priority_queue = [(0, K)]
    visited = {}
    while priority_queue:
        time, node = heappop(priority_queue)
        if node not in visited:
            visited[node] = time
            for v, w in g[node]:
                heappush(priority_queue, (time + w, v))
    return max(visited.values()) if len(visited) == N else -1


# print(networkDelayTime([[2,1,1],[2,3,1],[3,4,1]], 4, 2))
# print(networkDelayTime([[1,2,1],[2,3,2],[1,3,4]], 3, 1))
# print(networkDelayTime([[1,2,1]], 2, 2))
# print(networkDelayTime([[1,2,1],[2,3,7],[1,3,4],[2,1,2]], 3, 1))
