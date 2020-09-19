from typing import List, Dict
from collections import defaultdict

def checkValidString2(s: str) -> bool:
    if s == "":
        return True
    l = list(s)
    d = len(l)
    options = ["(", ")", ""]
    def solve(l, start, count):
        for i in range(start, d):
            if count < 0 or count > d - i:
                return False
            if l[i]  == "":
                continue
            elif l[i]  == "(":
                count += 1
            elif l[i]  == ")":
                count -= 1
            if l[i]  == "*":
                for j in options:
                    l[i] = j
                    if solve(l, i, count):
                        return True
                l[i] = "*"
        if count == 0:
            return True
        return False

    return solve(l, 0, 0)

def checkValidString(s: str) -> bool:
    # solve it with 2 stacks
    p_stack = []
    s_stack = []
    for i, c in enumerate(s):
        if c == "(":
            p_stack.append(i)
        elif c == ")":
            if p_stack:
                p_stack.pop()
            else:
                if s_stack:
                    s_stack.pop()
                else:
                    return False
        else:
            s_stack.append(i)
    while p_stack:
        c = p_stack.pop()
        if s_stack:
            s = s_stack.pop()
        else:
            return False
        if c > s:
            return False
    return True


# print(checkValidString("(())((())()()(*)(*()(())())())()()((()())((()))(*"))
# print(checkValidString("()"))
# print(checkValidString("("))
# print(checkValidString("())"))
# print(checkValidString("(())"))
# print(checkValidString("((*(*))"))


def networkDelayTime2(times: List[List[int]], N: int, K: int) -> int:
    # Dijikstra
    from heapq import heappush, heappop
    G = defaultdict(list)
    for u,v,w in times:
        G[u].append((w, v))
    seen = {}
    q = [(0, K)]
    while q:
        w, n = heappop(q)
        if n not in seen:
            seen[n] = w
            for weight, node in G[n]:
                heappush(q, (w + weight, node))
    return max(seen.values()) if len(seen) == N else -1


def networkDelayTime(times: List[List[int]], N: int, K: int) -> int:
    # simple dfs
    G = defaultdict(list)
    for u,v,w in times:
        G[u].append((w, v))

    distances = {node: float("inf") for node in range(1,N+1)}

    def dfs(node, time):
        if time >= distances[node]: return
        distances[node] = time
        for w, n in sorted(G[node]):
            dfs(n, time + w)

    dfs(K, 0)
    result = max(distances.values())
    return result if result < float("inf") else -1


#print(networkDelayTime([[2,1,1],[2,3,1],[3,4,1]], 4, 2))


def canJump(nums: List[int]) -> bool:
    if len(nums) < 2:
        return True
    i = 0
    cur = nums[i]
    while cur:
        if cur + i >= len(nums) - 1:
            return True
        i += 1
        cur -= 1
        cur = max(cur, nums[i])
    return False

# print(canJump([1,1,1,0]))
# print(canJump([3,2,1,0,4]))
# print(canJump([0,2,3]))

def subarraySum(nums: List[int], k: int) -> int:
    # sliding window doens't work with negative integers.
    cur = res = 0
    cache = {0: 1}
    for v in nums:
        cur += v
        rest = cache.get(cur - k, 0)
        res += rest
        add = cache.get(cur, 0) + 1
        cache[cur] = add

    return res

#
# print(subarraySum([-1,-1,1], 0))
# print(subarraySum([1,1,1], 2))
# print(subarraySum([1,2,1], 2))
# print(subarraySum([1,2,1,0], 3))
# print(subarraySum([1,2,-1,0], 3))
# print(subarraySum([1,1,2], 2))

def printAllSubArrays(nums: List[int]):
    n = len(nums)
    for i in range(n):
        for j in range(i, n+1):
            temp = []
            for k in range(i, j):
                temp.append(nums[k])
            if temp:
                print(temp)


#print(printAllSubArrays([1,2,3]))

def printAllSubArraysRec(nums):

    def rec(start, nums):
        if start >= len(nums):
            return
        result = []
        for i in range(start, len(nums)):
            result.append(nums[i])
            print(result)
        rec(start+1, nums)

    rec(0, nums)

#print(printAllSubArraysRec([1,2,3]))


