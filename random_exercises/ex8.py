from typing import List, Dict
from collections import defaultdict
from heapq import heappush, heappop

def longestOnes(A: List[int], K: int) -> int:
    # Given an array A of 0s and 1s, we may change up to K values from 0 to 1.
    # Return the length of the longest (contiguous) subarray that contains only 1s.
    j = 0
    for i in range(len(A)):
        K += A[i] - 1
        if K < 0:
            K -= A[j] - 1
            j += 1

    return i - j + 1


#print(longestOnes([1,1], 2)) # 2
# print(longestOnes([1,1,1,0,0,0,1,1,1,1,0], 2)) # 6
# print(longestOnes([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3)) # 10



def totalFruit(tree: List[int]) -> int:
    # find the longest contiguos array that contains only 2 distinct numbers
    j = 0
    cur = {}
    for i in range(len(tree)):
        cur[tree[i]] = cur.get(tree[i], 0) + 1
        if len(cur.keys()) > 2:
            if cur[tree[j]] == 1:
                cur.pop(tree[j])
            else:
                cur[tree[j]] = cur.get(tree[j]) - 1
            j += 1

    return len(tree) - j
#
# print(totalFruit([1,2,1])) # 3
# print(totalFruit([0,1,2,2])) # 3
# print(totalFruit([3,3,3,1,2,1,1,2,3,3,4])) # 5
# print(totalFruit([1,2,3,2,2])) # 4
# print(totalFruit([1,0,0,4,7,4,4,4,7])) # 4

def balancedString(s: str) -> int:
    # You are given a string containing only 4 kinds of characters 'Q', 'W', 'E' and 'R'.
    # minimum length of string that replaces the substring to be balanced
    seen = {"Q": 0, "W": 0, "E":0, "R": 0}
    for i in range(len(s)):
        seen[s[i]] += 1
    balanced = len(s) // 4
    if max([seen[x] for x in "QWER"]) <= balanced:
        return 0
    j = 0
    ans = float("Inf")
    for i in range(len(s)):
        seen[s[i]] -= 1
        while j <= i and max([seen[x] for x in "QWER"]) <= balanced:
            ans = min(ans, i-j + 1)
            seen[s[j]] += 1
            j += 1

    return ans

# print(balancedString("WWEQERQWQWWRWWERQWEQ")) # 4
# print(balancedString("QWER")) # 0
# print(balancedString("QWQQEEER")) #2
# print(balancedString("QQER")) #1
# print(balancedString("QQQR")) #2
# print(balancedString("QQQQ")) #3


def countAndSay(n: int) -> str:
    #https://leetcode.com/problems/count-and-say/
    def rec(x, k):
        if k == n:
            return x
        temp = ""
        j = 1
        i = 0
        while i < len(x):
            if i + 1 < len(x) and x[i] == x[i+1]:
                j += 1
            else:
                temp += str(j)
                temp += x[i]
                j = 1
            i += 1

        return rec(temp, k + 1)

    return rec("1", 1)

# print(countAndSay(1)) # 1
# print(countAndSay(2)) # 11
# print(countAndSay(3)) # 21
# print(countAndSay(4)) # 1211