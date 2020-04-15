from typing import List, Dict
from collections import defaultdict

def longestPalindrome(s: str) -> str:
    j = len(s)

    def look_left_right(l, r):
        while l >= 0 and r < j:
            if s[l] == s[r]:
                l -= 1
                r += 1
            else:
                return l+1, r-1, r-l-2
        return l+1, r-1, r-l-2

    result = (0,0,0)
    for i in range(len(s)):
        res = look_left_right(i-1,i+1)
        if res[2] > result[2]:
            result = res
        res = look_left_right(i, i+1)
        if res[2] > result[2]:
            result = res

    return s[result[0]:result[1]+1]

# print(longestPalindrome(""))
# print(longestPalindrome("acc"))
# print(longestPalindrome("babad"))
# print(longestPalindrome("adam"))
# print(longestPalindrome("addam"))

def isHappy(n: int) -> bool:
    seen = set()
    def calculate(n):
        n = str(n)
        res = 0
        for i in n:
            res += int(i)**2
        if res == 1:
            return True
        else:
            if res in seen:
                return False
            else:
                seen.add(res)
                return calculate(res)

    return calculate(n)

# print(isHappy(19))

def maxSubArray(nums: List[int]) -> int:
    maxC = res = nums[0]
    for i in range(1, len(nums)):
        maxC = max(nums[i], maxC + nums[i])
        res = max(maxC, res)

    return res

#print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))


def countElements(arr: List[int]) -> int:
    count = 0
    seen = set()
    for x in arr:
        seen.add(x)
    for x in arr:
        if x+1 in seen:
            count += 1

    return count

# print(countElements([1,2,3]))
# print(countElements([1,1,3,3,5,5,7,7]))


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    res = defaultdict(list)
    for item in strs:
        alpha = tuple(sorted(item))
        res[alpha].append(item)

    return res.values()


#print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))


def backspaceCompare(S: str, T: str) -> bool:
    i = len(S) - 1
    j = len(T) - 1
    count_j = 0
    count_i = 0
    while i >= 0 and j >= 0:
        while count_i and i >= 0 and S[i] != "#":
            i -= 1
            count_i -= 1
        while count_j and j >= 0 and T[j] != "#":
            j -= 1
            count_j -= 1
        if S[i] == "#":
            while i >= 0 and S[i] == "#":
                i -= 1
                count_i += 1
            continue
        if T[j] == "#":
            while j >= 0 and T[j] == "#":
                j -= 1
                count_j += 1
            continue
        if i >= 0 and j >= 0:
            if S[i] == T[j]:
                i -= 1
                j -= 1
                continue
            else:
                return False

    if i >=0:
        while i >= 0:
            while count_i and i >= 0 and S[i] != "#":
                i -= 1
                count_i -= 1
            if S[i] == "#":
                while i >= 0 and S[i] == "#":
                    i -= 1
                    count_i += 1
                continue
            return S[:i+1] == ""

    if j >=0:
        while j >= 0:
            while count_j and j >= 0 and T[j] != "#":
                j -= 1
                count_j -= 1
            if T[j] == "#":
                while j >= 0 and T[j] == "#":
                    j -= 1
                    count_j += 1
                continue
            return T[:j+1] == ""

    return True


def backspaceCompare2(S: str, T: str) -> bool:
    def build(s):
        stack = []
        for l in s:
            if l != "#":
                stack.append(l)
            elif stack:
                stack.pop()
        return stack

    return build(S) == build(T)



print(backspaceCompare2("j##yc##bs#srqpfzantto###########i#mwb", "j##yc##bs#srqpf#zantto###########i#mwb"))
print(backspaceCompare2("y#fo##f", "y#fx#o##f"))
print(backspaceCompare2("bbbextm", "bbb#extm"))
print(backspaceCompare2("ab#c", "ad#c"))
print(backspaceCompare2("ab##", "ll"))
print(backspaceCompare2("a#c", "b"))
print(backspaceCompare2("baa##", "b"))
print(backspaceCompare2("xywrrmp", "xywrrmu#p"))
print(backspaceCompare2("bxj##tw", "bxo#j##tw"))


