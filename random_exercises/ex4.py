from typing import List, Dict
from collections import defaultdict

def subarraySum(nums: List[int], k: int) -> int:
    # sliding window doesn't work because of negative numbers
    # sum(i, j) == sum(0, j) - sum(0, i)
    # we calculate the prefix sum (cumulative sum)
    # we're gonna use a hashtable to store the sum at all points
    # if we find that the sum - k = a number previously calculated, then we have a sum = k
    sums = {0: 1}
    result = 0
    cur = 0
    for i in range(len(nums)):
        cur += nums[i]
        result += sums.get(cur - k, 0)
        sums[cur] = sums.get(cur, 0) + 1

    return result

# print(subarraySum([1,1,1], 2))
# print(subarraySum([1,0,1], 2))
# print(subarraySum([1,-1,2, 2], 2))

def networkDelayTime(times: List[List[int]], N: int, K: int) -> int:
    # we can use dijistra algo
    from heapq import heappop, heappush
    g = defaultdict(list)
    for s, t, w in times:
        g[s].append((w, t))
    q = [(0, K)]
    seen = {}
    while q:
        w, t = heappop(q)
        if t not in seen:
            seen[t] = w
            for n, t in g[t]:
                heappush(q, (w+n, t))
    return max(seen.values()) if len(seen) == N else -1


# print(networkDelayTime([[2,1,1],[2,3,1],[3,4,1]], 4, 2))

def checkValidString(s: str) -> bool:
    balanced = 0
    for c in s:
        if c == "(" or c == "*":
            balanced += 1
        elif c == ")":
            balanced -= 1
            if balanced < 0:
                return False

    balanced = 0
    for c in reversed(s):
        if c == ")" or c == "*":
            balanced += 1
        elif c == "(":
            balanced -= 1
            if balanced < 0:
                return False

    return True