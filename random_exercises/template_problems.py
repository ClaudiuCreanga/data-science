from typing import List, Dict
from collections import defaultdict
from heapq import heappush, heappop

def canJump(nums: List[int]) -> bool:
    pass

print(canJump([2,3,1,1,4])) # t
print(canJump([3,2,1,0,4])) # f
print(canJump([1,1,2,2,0,1,1])) # t


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def bstFromPreorder(preorder: List[int]) -> TreeNode:
    # 2 solutions
    pass

print(bstFromPreorder([8,5,1,7,10,12]))  #[8,5,10,1,7,null,12]


def subarraySum(nums: List[int], k: int) -> int:
    pass

print(subarraySum([1,1,1], 2)) # 2
print(subarraySum([1], 0)) # 0


def minSubArrayLen(s: int, nums: List[int]) -> int:
    # Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead
    pass

print(minSubArrayLen(7, [2,3,1,2,4,3])) # 2


def longestOnes(A: List[int], K: int) -> int:
    # Given an array A of 0s and 1s, we may change up to K values from 0 to 1.
    # Return the length of the longest (contiguous) subarray that contains only 1s.
    pass

print(longestOnes([1,1,1,0,0,0,1,1,1,1,0], 2)) # 6
print(longestOnes([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3)) # 10


def totalFruit(tree: List[int]) -> int:
    # find the longest contiguos array that contains only 2 distinct numbers
    pass

print(totalFruit([1,2,1])) # 3
print(totalFruit([0,1,2,2])) # 3
print(totalFruit([3,3,3,1,2,1,1,2,3,3,4])) # 5
print(totalFruit([1,2,3,2,2])) # 4


def balancedString(s: str) -> int:
    #You are given a string containing only 4 kinds of characters 'Q', 'W', 'E' and 'R'.
    # minimum length of string that replaces the substring to be balanced
    pass

print(balancedString("QWER")) #0
print(balancedString("QQER")) #1
print(balancedString("QQQR")) #2
print(balancedString("QQQQ")) #3


def countAndSay(n: int) -> str:
    #https://leetcode.com/problems/count-and-say/
    pass

print(countAndSay(1)) # 1
print(countAndSay(3)) # 21
print(countAndSay(4)) # 1211
