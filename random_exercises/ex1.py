from typing import List

def rotLeft(a, d):
    for _ in range(d):
        a.append(a[0])
        del a[0]

    return a

#print(rotLeft([1, 2, 3, 4, 5], 3))

def minimumBribes(q):
    result = 0
    flag = False
    for passing in range(len(q)-1, 0, -1):
        if flag:
            result = "Too chaotic"
            break
        bribed = 0
        for i in range(passing):
            if q[i] > q[i+1]:
                q[i], q[i+1] = q[i+1], q[i]
                result += 1
                bribed += 1
                if bribed > 2:
                    flag = True
                    break
            else:
                bribed = 0

    return result

#print(minimumBribes([5, 1, 2, 3, 7, 8, 6, 4]))


def bubblesort(q):
    for passing in range(len(q)-1, 0, -1):
        for i in range(passing):
            if q[i] > q[i+1]:
                q[i], q[i+1] = q[i+1], q[i]
    return q

#print(bubblesort([5,1,4,2,8]))

def two_strings(a, b):
    edits = 0
    if abs(len(b) - len(a)) > 1:
        return "the words don't match"
    for index, letter in enumerate(a):
        if index <= len(b) - 1:
            if b[index] != letter:
                edits += 1
                if len(b) - len(a) == 1:
                    b = b[:index] + b[index+1:]
                elif len(b) - len(a) == -1:
                    b = b[:index] + letter + b[index:]
        else:
            edits += 1
    if edits > 1:
        return "the words don't match"

    return "the words match"

# print(two_strings("abc", "abc"))
# print(two_strings("adc", "abc"))
# print(two_strings("abc", "adc"))
# print(two_strings("bbc", "adc"))
# print(two_strings("abc", "ddc"))
# print(two_strings("bc", "dbc"))
# print(two_strings("abc", "bc"))
# print(two_strings("abc", "c"))
# print(two_strings("c", "c"))
# print(two_strings("ac", "aac"))
# print(two_strings("ac", "aaac"))
# print(two_strings("aaaaac", "aaac"))
# print(two_strings("aaaaac", "aaaac"))
# print(two_strings("aaac", "aaaac"))
# print(two_strings("abcde", "accde"))
# print(two_strings("abcde", "abcdf"))
# print(two_strings("abcdef", "abcde"))
# print(two_strings("geek", "geeks"))
# print(two_strings("m", ""))
# print(two_strings("", "m"))

def find_two_strings(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    i = 0
    j = 0
    edits = 0
    while (i < len(a) and j < len(b)):
        if a[i] != b[j]:
            edits += 1
            if len(a) > len(b):
                i+=1
            elif len(b) < len(a):
                j+=1
            else:
                i += 1
                j += 1
        else:
            i += 1
            j += 1
    if i < len(a) or j < len(b):
        edits += 1

    return not edits > 1
#
# print(find_two_strings("abc", "abc"))
# print(find_two_strings("adc", "abc"))
# print(find_two_strings("abc", "adc"))
# print(find_two_strings("bbc", "adc"))
# print(find_two_strings("abc", "ddc"))
# print(find_two_strings("bc", "dbc"))
# print(find_two_strings("abc", "bc"))
# print(find_two_strings("abc", "c"))
# print(find_two_strings("c", "c"))
# print(find_two_strings("ac", "aac"))
# print(find_two_strings("ac", "aaac"))
# print(find_two_strings("aaaaac", "aaac"))
# print(find_two_strings("aaaaac", "aaaac"))
# print(find_two_strings("aaac", "aaaac"))
# print(find_two_strings("abcde", "accde"))
# print(find_two_strings("abcde", "abcdf"))
# print(find_two_strings("abcdef", "abcde"))
# print(find_two_strings("geek", "geeks"))
# print(find_two_strings("m", ""))
# print(find_two_strings("", "m"))

def build_spiral_array(a):
    result = []
    rows = len(a)
    columns = len(a[0])
    row_index = 0
    column_index = 0
    while row_index < rows and column_index < columns:

        for i in range(column_index, columns):
            result.append(a[row_index][i])

        row_index += 1

        for i in range(row_index, rows):
            result.append(a[i][columns - 1])

        columns -= 1

        if row_index < rows:
            for i in range(columns - 1, column_index - 1, -1):
                result.append(a[rows - 1][i])

            rows -= 1

        if column_index < columns:
            for i in range(rows - 1, row_index - 1, -1):
                result.append(a[i][column_index])

            column_index += 1

    return " ".join(map(str, result))

#
# print(build_spiral_array([[1,    2,   3,   4],[5,    6,   7,   8],[9,   10,  11,  12],[13,  14,  15,  16]]))
# print(build_spiral_array([ [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],[13, 14, 15, 16, 17, 18]]))
# print(build_spiral_array([ [1,2,3], [8,9,4], [7,6,5]]))


def look_and_say(n):
    a = {
        "1": "11",
        "11": "21",
        "2": "12",
        "111": "3",
        "22": "22",
        "3": "13"

    }
    result = []
    count = 0

    def rec(x, count):
        if count > n:
            return
        count += 1
        r = ""
        i = 0
        while i < len(x):
            if i + 1 < len(x):
                if x[i] == x[i+1]:
                    r += a[x[i] + x[i+1]]
                    i += 1
                else:
                    r += (a[x[i]])
            else:
                r += a[x[i]]

            i += 1

        result.append(r)
        rec(r, count)

    rec(["1"], count)
    return result

#print(look_and_say(9))


def countnndSay(n):
    if n == 1:
        return "1"
    if n == 2:
        return "11"

    s = "11"

    for i in range(3, n+1):
        count = 1
        s += "$"
        tmp = ""
        l = len(s)

        for j in range(1, l):
            current = s[j]
            previous = s[j-1]
            if current != previous:
                tmp += str(count) + previous
                count = 1
            else:
                count += 1

        s = tmp
    return s

#print(countnndSay(9))

def myregex(pattern, mystring):
    i = 0
    while i < len(mystring):
        if mystring[i] == pattern[i]:
            return myregex(pattern[i+1:], mystring[i+1:])
        elif pattern[i] == ".":
            return myregex(pattern[i+1:], mystring[i+1:])
        else:
            return False
    return True




#print(myregex("dasss","dasss"))

def bonetrousle(n, k, b):
    minValue = b * (b + 1) / 2
    maxValue = b * (2 * k - b + 1)/ 2
    if minValue > n or maxValue < n:
        return [-1]
    if b == 1:
        return [n]
    result = list(range(1, b+1))
    r = int((n-minValue) % b)
    q = int((n -minValue) // b)

    for index, item in enumerate(result):
        result[index] += q
        if index > len(result) - r - 1 and index <= len(result) - 1:
            result[index] += 1
    print(len(set(result)) == b)
    return result



# print(bonetrousle(12, 8, 3))
# print(bonetrousle(10, 3, 3))
# print(bonetrousle(22,7,6))
# print(bonetrousle(38, 10, 7))
# print(bonetrousle(809880023823305331, 906161411045895284, 52920))


def bonetrousle2(n, k, b):
    init = b * (b + 1) // 2
    extra = (n - init) // b + 1
    over = init + extra * b - n
    answer = list(range(1, b + 1))
    answer = [i + extra - 1 for i in answer[:over]] + [i + extra for i in answer[over:]]
    if answer[-1] <= k and answer[0] >= 1:
        return answer
    else:
        return [-1]



#print(bonetrousle2(12, 8, 3))


def reverseStr(s: str, k: int) -> str:
    result = ""
    reverse = True
    i = 0
    while i < len(s):
        temp = s[i:i + k]
        if reverse:
            result += temp[::-1]
        else:
            result += temp
        reverse = not reverse
        i += k

    return result


#print(reverseStr("abcdefg", 2))

def isPalindrome(x: int) -> bool:
    to_str = str(int)
    reversed_str = ""
    i = len(to_str) - 1
    while i >= 0:
        reversed_str += to_str[i]
        i -= 1
    return reversed_str == to_str

def isPalindrome2( x: int) -> bool:
    to_str = str(x)
    reversed_str = to_str[::-1]
    return reversed_str == to_str

#print(isPalindrome(121))

def reverseStringConstantSPace(s): # strings are immutable in python so it's not constant space
    s = [i for i in s]
    start = 0
    end = len(s) - 1
    while start < end:
        s[start], s[end] = s[end], s[start]
        start += 1
        end += -1
    return "".join(s)

# rint(reverseStringConstantSPace("abcqdef"))

def isPalindromeInt(x: int) -> bool:
    y = 0
    z = x
    while x > 0:
        last_number = x % 10
        x = x // 10
        y += last_number
        y *= 10
    y = y // 10
    return y == z

#print(isPalindromeInt(121))

def removeDuplicates( nums: List[int]) -> int:
    i = 1
    while i < len(nums):
        if nums[i] == nums[i-1]:
            nums.pop(i)
        else:
            i += 1

    return len(nums)

#print(removeDuplicates([1,1,2]))

def removeDuplicates2(nums):
    if len(nums) == 0:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if (nums[j] != nums[i]):
            i += 1
            nums[i] = nums[j]
    return i + 1

#print(removeDuplicates2([1,1,2]))

def maxProfit(prices: List[int]) -> int:
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit

#print(maxProfit([7, 1, 5, 3, 6, 4]))


def permutations(string, step = 0):

    # if we've gotten to the end, print the permutation
    if step == len(string):
        print("".join(string))

    # everything to the right of step has not been swapped yet
    for i in range(step, len(string)):

        # copy the string (store as array)
        string_copy = [character for character in string]

        # swap the current index with the step
        string_copy[step], string_copy[i] = string_copy[i], string_copy[step]

        # recurse on the portion of the string that has not been swapped yet (now it's index will begin with step + 1)
        permutations(string_copy, step + 1)

# print(permutations("ABCD"))

def isValid(s: str) -> bool:
    close = {
        ")": "(",
        "]": "[",
        "}": "{"
    }
    if s == "":
        return True

    stack = []
    for x in range(len(s)):
        if s[x] == "(" or s[x] == "{" or s[x] == "[":
            stack.append(s[x])
        else:
            if not len(stack):
                return False
            last_item = stack.pop()
            if close[s[x]] != last_item:
                return False
    return len(stack) == 0

#print(isValid("]"))

def twoSuma(a, t):
    seen = {}
    result = []
    for i in range(len(a)):
        temp = t - a[i]
        if temp not in seen:
            seen[a[i]] = i
        else:
            result.append((seen[temp], i))
    return result

# print(twoSuma([2, 7, 11, 15, 8, 6, 1],9))

def checkTwoSOneEditAway(A:str, B:str) -> bool:
    if abs(len(A) - len(B)) > 1:
        return False
    for i in range(len(A)):
        if i < len(B):
            if A[i] != B[i]:
                return A[i+1:] == B[i:] or A[i+1:] == B[i+1:] or A[i] == B[i+1:]

    return True
#
# print(checkTwoSOneEditAway("pale", "ple"))
# print(checkTwoSOneEditAway("pales", "pale"))
# print(checkTwoSOneEditAway("pale", "bale"))
# print(checkTwoSOneEditAway("pale", "bae"))
# print(checkTwoSOneEditAway("pale", "pales"))
# print(checkTwoSOneEditAway("palas", "pale"))
# print(checkTwoSOneEditAway("palasaa", "palas"))
# print(checkTwoSOneEditAway("", "a"))
# print(checkTwoSOneEditAway("a", ""))
# print(checkTwoSOneEditAway("palas", "palas"))

def compress(s):
    comp = []
    if len(s):
        count = 1
        current = s[0]
        i = 1
        while i < len(s):
            if s[i] != current:
                comp.append(current)
                comp.append(count)
                current = s[i]
                count = 1
            else:
                count += 1
            i += 1
        comp.append(current)
        comp.append(count)
    s2 = "".join(map(str,comp))

    return s if len(s) <= len(s2) else s2

print(compress("aaabccccc"))
print(compress(""))

# anagrams and palindrom