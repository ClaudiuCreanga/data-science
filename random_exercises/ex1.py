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

# print(reverseStringConstantSPace("abcqdef"))

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

#print(permutations("ABCD"))

def permutations2(s):

    def perm(s, l, r):
        if l == r:
            print("".join(s))

        for i in range(l, r+1):
            s[l], s[i] = s[i], s[l]
            perm(s, l+1, r)
            s[l], s[i] = s[i], s[l]


    return perm(list(s), 0, len(s) - 1)

# print(permutations2("ABCD"))

def permute3(nums: List[int]) -> List[List[int]]:
    result = []

    def perm(a, k=0):
        if k == len(a):
            result.append([b for b in a])
        else:
            for i in range(k, len(a)):
                a[k], a[i] = a[i], a[k]
                perm(a, k + 1)
                a[k], a[i] = a[i], a[k]  # backtrack or else do the modification in a copy of the list

    perm(nums)
    return result

# print(permute3([0,1,2]))

def findIfpermutationIsPalindrom(s):
    v = {}
    for c in s:
        if c in v:
            v[c] = not v[c]
        else:
            v[c] = False

    middle = False
    for a,b in v.items():
        if not b:
            if not middle:
                middle = True
            else:
                return False

    return True

#print(findIfpermutationIsPalindrom("tactcoaa"))


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
        temp = t - a[i].lower()
        if temp not in seen:
            seen[a[i]] = i
        else:
            result.append((seen[temp], i))
    return result

# print(twoSuma([2, 7, 11, 15, 8, 6, 1],9))

def isPalindrome3( s: str) -> bool:
    j = len(s) - 1
    i = 0
    while i < len(s) // 2:
        if not s[i].isalnum():
            i += 1
            continue
        if not s[j].isalnum():
            j -= 1
            continue
        if s[i].lower() != s[j].lower():
            return False
        j -= 1
        i += 1
    return True

#print(isPalindrome3("A man, a plan, a canal: Panama"))


def validPalindrome5(s: str) -> bool:
    i, j, edited = 0, len(s) - 1, False
    while i < j:
        if s[i] != s[j]:
            if edited:
                return False
            edited = True
            i += 1
        i += 1
        j -= 1
    return True

def validPalindrome4(s: str) -> bool:
    L = len(s)
    for i in range(L//2+1):
        if (s[i] != s[-i-1]):
            return s[i+1:L-i] == s[i+1:L-i][::-1] or s[i:L-i-1] == s[i:L-i-1][::-1]
    return True

def validPalindromeRec(s):

    def rec(start = 0, end = len(s) - 1, skipOnce = False):
        while start < end:
            if s[start] != s[end]:
                if not skipOnce:
                    return rec(start + 1, end, True) or rec(start, end - 1, True)
                else:
                    return False
            start += 1
            end -= 1
        return True

    return rec()

def validPalindromeWhy(s):
    start = 0
    end = len(s) - 1
    while start < end:
        if s[start] != s[end]:
            return s[start + 1:len(s) // 2] == s[len(s)//2 + 1: end + 1][::-1] \
                   or s[start: len(s)//2] == s[len(s)//2: end][::-1]
        start += 1
        end -= 1

    return True

# print(validPalindromeWhy("abba"))
# print(validPalindromeWhy("abcba"))
# print(validPalindromeWhy("ebcdba"))
# print(validPalindromeWhy("abcdedcba"))
# print(validPalindromeWhy("abcdedcb"))

def searchBinary(nums: List[int], target: int) -> int:
    def search(a, l, r):
        if r >= l:
            middle = (l + r) // 2
            if a[middle] == target:
                return middle
            elif a[middle] < target:
                return search(a, middle + 1, r)
            else:
                return search(a, l, middle - 1)
        else:
            return (r + l) // 2 + 1

    return search(nums, 0, len(nums) - 1)

def searchBinaryItirative(nums, target):
    l = 0
    r = len(nums) - 1
    while l <= r:
        middle = (l + r) // 2
        if nums[middle] == target:
            return middle
        elif nums[middle] < target:
            l = middle + 1
        else:
            r = middle - 1
    return -1

def searchBinaryLoHi(nums, target):
    low = 0
    hi = len(nums) - 1
    while low <= hi:
        middle = (low + hi) // 2 # low + (hi - lo) // 2 to avoid overflow
        if nums[middle] == target:
            return middle
        elif nums[middle] < target:
            low = middle + 1
        else:
            hi = middle - 1
    return (low + hi) // 2 + 1

# print(searchBinaryLoHi([1,3,5,6, 8, 9, 12], 2))

# print(searchBinaryItirative([1,3,5,6], 6))
# print(searchBinary([1,3,5,6, 8, 9, 10, 12], 10))
# print(searchBinary([1,3,5,6, 8, 9, 10, 12, 13], 13))
# print(searchBinary([1,3,5,6, 8, 9, 10, 12, 13], 1))
# print(searchBinary([1,3,5,6, 8, 9, 10, 12, 13], 8))
# print(searchBinary([1,3,5,6, 8, 9, 10, 12, 13], 90))

def findRadius2(houses: List[int], heaters: List[int]) -> int: # O: HeLogHe + HoLogHe
    heaters.sort()
    radius = 0
    def binary_search(l, r, t):
        if r >= l:
            middle = (l + r) // 2
            if heaters[middle] == t:
                return False
            elif heaters[middle] < t:
                return binary_search(middle + 1, r, t)
            else:
                return binary_search(l, middle - 1, t)
        else:
            return (r + l) // 2 + 1

    for i in range(len(houses)):
        item = binary_search(0, len(heaters) - 1, houses[i])
        if item is not False:
            if item < len(heaters) - 1:
                hi = heaters[item]
            else:
                hi = heaters[-1]
            if item > 0:
                low = heaters[item - 1]
            else:
                low = heaters[0]
            new_radius = min([abs(houses[i] - low), abs(hi - houses[i])])
            radius = max(new_radius, radius)

    return radius

# print(findRadius2([1,5], [10]))
# print(findRadius2([1,2,3,4], [1,4]))
# print(findRadius2([1,2,3],[2]))
# print(findRadius2([1,2,3,4,5,6,7],[2]))
# print(findRadius2([1,2,3,4,5,6,7],[2,5]))
# print(findRadius2([1,2,3,4,5,6],[2,5]))
# print(findRadius2([2,3,1,4,6,5],[5, 2]))
# print(findRadius2([1,2,3],[1,2,3]))
# print(findRadius2([999,999,999,999,999],[499,500,500,501]))

# anagrams and palindrom

def checkTwoSOneEditAway(A:str, B:str) -> bool:
    if abs(len(A) - len(B)) > 1:
        return False
    for i in range(len(A)):
        if i < len(B):
            if A[i] != B[i]:
                return A[i+1:] == B[i:] or A[i+1:] == B[i+1:] or A[i] == B[i+1:]

    return True

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

# print(compress("aaabccccc"))
# print(compress(""))

def rotate(M):
    result = [[0, 0, 0] for x in range(len(M))]
    column = len(M)
    for i in range(len(M)):
        row = 0
        column -= 1
        for j in range(len(M)):
            result[row][column] = M[i][j]
            row += 1
    return result

# print(rotate([[1,2,3], [6,7,8], [9,4,2]]))

def rotate2(M):
    n = len(M)
    for i in range(n // 2):
        first = i
        last = n - 1 - i
        for i in range(i, last):
            offset = i - first
            top = M[first][i]
            M[first][i] = M[last-offset][first]
            M[last-offset][first] = M[last][last-offset]
            M[last][last-offset] = M[i][last]
            M[i][last] = top

    return M
# print(rotate2([[1,2,3], [6,7,8], [9,4,2]]))

def zero(M):
    row, col = [], []
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] == 0:
                row.append(i)
                col.append(j)
    for i in range(len(row)):
        for j in range(len(M[0])):
            M[row[i]][j] = 0
        for x in range(len(M)):
            M[x][col[i]] = 0
    return M

# print(zero([[1,1,1,1,1,1,1,1,1], [0,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1]]))

def rotateS(s1, s2):
    i = 0
    while i < len(s2):
        temp = s2[0]
        s2 = s2[1:]
        s2 += temp
        if s2 == s1:
            return True
        i += 1
    return False

def rotateS2(s1, s2):
    for i in range(len(s2)):
        s2 += s2[i]
        if s1 in s2:
            return True
    return False


def rotateS3(s1, s2):
    s1 += s1
    if s2 in s1:
        return True
    return False

#print(rotateS3("waterbottle", "erbottlewat"))

def calendarConflict(cal):
    conflicts = []
    temp_conflicts = [cal[0][2]]
    end = cal[0][1]
    for i in range(1, len(cal)):
        if cal[i][0] >= end:
            if len(temp_conflicts) > 1:
                conflicts.append(temp_conflicts)
            temp_conflicts = []
        end = max(end, cal[i][1])
        temp_conflicts.append(cal[i][2])
    if len(temp_conflicts) > 1:
        conflicts.append(temp_conflicts)

    return conflicts

#print(calendarConflict([[1,2,"a"], [2,4, "b"], [3,5, "c"], [7,9, "d"]]))

class StackMin():

    def __init__(self):
        self.data = []
        self.current_min = None

    def push(self, d):
        if self.current_min:
            self.data.append((d, self.current_min))
            if d < self.current_min:
                self.current_min = d
        else:
            self.current_min = d
            self.data.append((d, None))

    def remove(self):
        item = self.data.pop()
        if item[0] == self.current_min:
            self.current_min = item[1]

# da = StackMin()
# print(da.current_min)
# da.push(3)
# print(da.current_min)
# da.push(2)
# print(da.current_min)
# da.push(7)
# print(da.current_min)
# da.push(1)
# print(da.current_min)
# da.remove()
# print(da.current_min)
# da.remove()
# print(da.current_min)
# da.remove()
# print(da.current_min)
# da.remove()
# print(da.current_min)


def SortStack(l):

    l2 = []
    temp = []
    while len(l) != len(l2):
        min = float("inf")
        while len(l):
            item = l.pop()
            if item < min:
                temp.append(min)
                min = item
            else:
                temp.append(item)

        l2.append(min)

        while len(temp):
            l.append(temp.pop())

    return l2

#print(SortStack([3,4,1,2,5,2]))

def coin_change(amount, coins):
    combinations = [0 for x in range(amount + 1)]
    combinations[0] = 1
    for coin in coins:
        for x in range(1, len(combinations)):
            if x >= coin:
                combinations[x] += combinations[x-coin] # so the trick is to take the previous row value + the value from the column - coin
    return combinations[amount]

# print(coin_change(12, [1,2,5]))

def KnapsackProblem(weight, items):
    combinations = [0 for x in range(weight + 1)]
    print(combinations)
    for w,v in items.items():
        for x in range(len(combinations)):
            if w <= x:
                combinations[x] = max(combinations[x], combinations[x - w] + v)

    return combinations[-1]

# print(KnapsackProblem(5, {5: 60, 3: 50, 4: 70, 2: 30}))

def levenstein_distance(a,b): # we need to use a matrix, can't do it with a single row
    combinations = [[0 for x in range(len(a) + 1)] for i in range(len(b) + 1)]
    combinations[0] = [x for x in range(len(a) + 1)]
    print(combinations[0])
    for i in range(1, len(b) + 1):
        for j in range(len(a) + 1):
            if j == 0:
                edits = combinations[i-1][0] + 1
            else:
                edits = min(combinations[i][j-1], combinations[i-1][j-1], combinations[i-1][j])
                if a[j-1] != b[i-1]:
                    edits += 1
            combinations[i][j] = edits

        print(combinations[i])

    return combinations[len(b)][len(a)]
#print(levenstein_distance("benyam", "ephrem"))
from collections import defaultdict


class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Push current vertex to stack which stores result
        stack.insert(0, v)

    def topological_sort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Print contents of stack
        return stack

g= Graph(6)
g.addEdge(5, 2)
g.addEdge(5, 0)
g.addEdge(4, 0)
g.addEdge(4, 1)
g.addEdge(2, 3)
g.addEdge(3, 1)
#print(g.topological_sort())
graph = {
    'a': ['b', 'c'],
    'b': ['d'],
    'c': ['d'],
    'd': ['e'],
    'e': []
}
def iterative_topological_sort(graph, start):
    seen = set()
    stack = []    # path variable is gone, stack and order are new
    order = []    # order will be in reverse order at first
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v)
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]: # check that the current value is not a dependence of the last item in stack
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]

def recursive_topological_sort(graph, node):
    result = []
    seen = set()

    def recursive_helper(node):
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                recursive_helper(neighbor)
        result.append(node)

    recursive_helper(node)
    return result
#print(recursive_topological_sort(graph, "a"))


def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    '''
    Use defaultdict to build a graph based on prerequisites information, try to find a loop in this graph
    Assign 3 states for each course 0, 1, 2. 0 for not visited, 1 for visiting and 2 for visited
    During DFS search, if we meet a course that with state == 1, then there is a loop. If we meet a course with state == 0, call DFS on new course.
    At each course, the graphy stores all possible next move (prerequisites).
    Time: O(n)
    Space: O(n)
    '''
    def DFS(start, my_dict, course_state):
        course_state[start] = 1
        for pre_course in my_dict[start]:
            if course_state[pre_course] == 1:
                return True
            if course_state[pre_course] == 0:
                if DFS(pre_course, my_dict, course_state):
                    return True
        course_state[start] = 2
        return False

    if not numCourses or not prerequisites:
        return True  # Assume no course to take returns True

    my_dict = defaultdict(list)
    for p in prerequisites:
        my_dict[p[0]].append(p[1])

    # Init states for all courses
    course_state = [0] * numCourses

    for n in range(numCourses):
        if course_state[n] == 0:  # Call DFS from this node and look for a loop
            loop = DFS(n, my_dict, course_state)
            if loop:
                return False
    return True
# analyze this one

fiblist = [0,1]
def fib_dp(n):
    if n<0:
        print("Incorect")
    if n <= len(fiblist):
        return fiblist[n-1]
    else:
        temp = fib_dp(n-1) + fib_dp(n-2)
        fiblist.append(temp)
        return temp

#print(fib_dp(8))

def threeSum(nums: List[int]) -> List[List[int]]:

    # return empty list if list size < 3
    if not nums or len(nums) < 3:
        return []

    nums.sort()
    # if the smallest number >0 or largest < 0, we can't have a sum of 0
    if nums[0] > 0 or nums[-1] < 0:
        return []

    res = []

    # find the index of the first non negative element
    p_non_neg_index = next(index for index, val in enumerate(nums) if val >= 0)

    # we will keep the pivot index to 0 and iterate till p_non_neg_index
    for pivot in range(p_non_neg_index + 1):

        # skip already computed pivots
        if pivot > 0 and nums[pivot - 1] == nums[pivot]:
            continue

        left = pivot + 1
        right = len(nums) - 1

        while left < right:
            s = nums[pivot] + nums[left] + nums[right]
            if s > 0:
                # if sum greater than zero we will move the right pointer towards the left(smaller value)
                # if the previous element is same , we will skip the that element
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                right -= 1
            elif s < 0:
                # if sum less than zero we will move the left pointer towards the right(higer value)
                # if the next element is same , we will skip the that element
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                left += 1
            else:

                # if sum is zero, we will append the triplet to list
                # move both left and right to next position avoiding the duplicates
                res.append([nums[pivot], nums[left], nums[right]])
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                right -= 1
                left += 1
    return res


def floodFill(image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
    r = len(image)
    c = len(image[0])
    original_color = image[sr][sc]
    visited = set()

    def dfs(x, y):
        if x < 0 or y < 0:
            return
        if x >= r or y >=c:
            return

        if image[x][y] == original_color:
            image[x][y] = newColor
            if (x,y) not in visited:
                visited.add((x,y))
                dfs(x-1, y)
                dfs(x+1, y)
                dfs(x, y-1)
                dfs(x, y+1)

    dfs(sr, sc)
    return image


#print(floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2))

def articulation_point():
    pass

def triple_step(n):
    """how many possible ways you can run up the stairs n if you can hop 1,2 or 3 at a time. O(3**n)"""
    if n < 0:
        return 0
    if n == 0:
        return 1
    return triple_step(n-1) + triple_step(n-2) + triple_step(n-3)

items = []
def triple_step_dp(n):
    """how many possible ways you can run up the stairs n if you can hop 1,2 or 3 at a time."""
    if n < 0:
        return 0
    if n == 0:
        return 1
    if n <= len(items):
        return items[n-1]
    else:
        temp = triple_step_dp(n-1) + triple_step_dp(n-2) + triple_step_dp(n-3)
        items.append(temp)
        return temp

#print(triple_step_dp(5))

def robot_grid(grid):
    """exponential big O. O(2**r+c) because each point has 2 possibilities"""
    success = False

    def rec(current):
        nonlocal success
        if success:
            return
        r = current[0]
        c = current[1]
        if current == (0,0):
            success = True
            return

        if grid[r][c] != 'X' and r >= 0 and c >= 0:
            rec((r-1, c))
            rec((r, c-1))

    rec((len(grid)-1,len(grid[0]) - 1))

    return success

def robot_grid_dp(grid): # do this to save the good path
    """Big O r*c because we don't visit any cells twice"""
    success = False
    visited = set()

    def rec(current):
        nonlocal success
        if success or current in visited:
            return
        r = current[0]
        c = current[1]
        if current == (0,0):
            success = True
            return

        if grid[r][c] != 'X' and r >= 0 and c >= 0:
            visited.add((r,c))
            rec((r-1, c))
            rec((r, c-1))

    rec((len(grid)-1,len(grid[0]) - 1))

    return success

# print(robot_grid_dp([[0, "X", 0, 0, 0],
#                       [0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0],
#                       ["X", 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0]]))

def magic_index(A):

    def binary_search(A, l, r):
        if l <= r:
            middle = l + (r-l) // 2

            if A[middle] == middle:
                return True
            elif A[middle] < middle:
                return binary_search(A, middle + 1, r)
            else:
                return binary_search(A, l, middle - 1)
        else:
            return False

    return binary_search(A, 0, len(A) - 1)

#print(magic_index([-2, -1, 1, 2, 5, 6]))

def magic_index_duplicates(A):
    def magic_fast(A, l, r):

        if l<=r:
            middle = l + (r-l) // 2
            if A[middle] == middle:
                return True

            left = magic_fast(A, l, min(middle - 1, A[middle]))
            if left >= 0:
                return left
            right = magic_fast(A, max(middle+1, A[middle]), r)
            return right
        else:
            return -1

    return magic_fast(A, 0, len(A) - 1)


#print(magic_index_duplicates([-10, -5, 2, 2, 2,3,4, 7, 9, 12]))
def subsets_set(S):
    '''all subssets of a set. here we must be careful to use a deep copy of lists, even the items in the list should be copied over'''
    def rec(S, result, index=0):
        if index < len(S):
            temp = list(result)
            item = S[index]
            for subset in temp:
                new = subset[:]
                new.append(item)
                result.append(new)
            return rec(S, result, index+1)
        else:
            return result

    return rec(S, [[]])

#print(subsets_set([1,2,3,4,5]))

def hanoi_tower(n, source, dest, util):
    if n == 1:
        print("Move disk 1 from rod", source, "to rod", dest)
        return
    hanoi_tower(n - 1, source, util, dest)
    print("Move disk", n, "from rod", source, "to rod", dest)
    hanoi_tower(n - 1, util, dest, source)

#print(hanoi_tower(4, "A", "C", "B"))

def permutations_without_dups(S, step = 0, result=[]):
    if step == len(S):
        result.append("".join(S))
    else:
        for i in range(step, len(S)):
            temp = [x for x in S]
            temp[i], temp[step] = temp[step], temp[i]
            permutations_without_dups(temp, step+1)

    return result

def permutations_without_dups2(S,l,r,result=[]):
    if l==r:
        result.append("".join(S))
    else:
        for i in range(l, r+1):
            S[i], S[l] = S[l], S[i]
            permutations_without_dups2(S, l+1, r, result)
            S[l], S[i] = S[i], S[l]


    return result

#print(permutations_without_dups2(list("abcd"), 0, len("abcd") - 1))

def all_valid_paran(n,result, current=2 ):
    if current <= n:
        new = set()
        for x in result:
            for i in range(len(x)):
                if x[i] == "(":
                    s = x[:i+1] + "()" + x[i+1:]
                    new.add(s)
            new.add("()" + x)
        return all_valid_paran(n, new, current+1)

    else:
        return set(result)
qresult = set()
qresult.add("()")

#print(all_valid_paran(3 ,qresult))

def paintFill(A, target, newcolor):
    existing_color = A[target[0]][target[1]]
    visited = set()
    def dfs(r,c):
        if r >= 0 and c >= 0 and r < len(A) and c < len(A):
            if A[r][c] == existing_color and (r,c) not in visited:
                A[r][c] = newcolor
                visited.add((r,c))
                dfs(r-1,c)
                dfs(r,c-1)
                dfs(r+1, c)
                dfs(r,c+1)
    dfs(target[0], target[1])
    return A

# print(paintFill([[1,1,1],
#                  [1,1,0],
#                  [1,0,1]], (1, 1), 2))

def coin_change2(n: int, coins: List[int]) -> int:
    combinations = [[0 for x in range(n + 1)] for x in range(len(coins) + 1)]
    combinations[0][0] = 1
    for r in range(len(combinations)):
        if r == 0:
            continue
        coin = coins[r - 1]
        for c in range(len(combinations[0])):
            if c == 0:
                combinations[r][c] = 1
            else:
                if c - coin >= 0:
                    combinations[r][c] = combinations[r-1][c] + combinations[r][c - coin]
                else:
                    combinations[r][c] = combinations[r - 1][c]

    return combinations[r][c]


def coin_change3(n, coins):
    pass

#print(coin_change3(50, [1, 5, 10, 25]))

def minimum_amount_coins(amount, coins):
    if amount < 0:
        return -1
    coins = sorted(coins)
    d = [amount + 1] * (amount + 1)
    d[0] = 0
    for i in range(amount + 1):
        for j in coins:
            if j <= i:
                d[i] = min(d[i], d[i - j] + 1)
            else:
                break
    return -1 if d[-1] > amount else d[-1]
#print(minimum_amount_coins(11, [5,2,1]))


def coinChange(coins: List[int], amount: int):
    minimum = float("Inf")

    def dfs(amount, changes):
        nonlocal minimum
        if amount == 0:
            minimum = min(changes, minimum)
        elif amount > 0:
            for coin in coins:
                dfs(amount - coin, changes+1)
        else:
            return

    dfs(amount, 0)
    if minimum == float("Inf"):
        return -1
    return minimum

#print(coinChange([5,2,1], 10))

def NQueen(n):

    def isValid(placements):
        row = len(placements) - 1
        for i in range(row):
            diff = abs(placements[i] - placements[row])
            if diff == 0 or diff == row - i:
                return False
        return True

    def solve(n, r, placements):
        if r == n:
            return placements
        else:
            for c in range(n):
                placements.append((r, c))
                if isValid(placements):
                    solve(n, r+1, placements)
                placements.pop()

    def driver(n):
        return solve(n, 0, [])

    return driver(n)

#print(NQueen(4))

def knap2(weight, items):
    combinations = [0 for x in range(weight + 1)]
    for item in items:
        for i in range(len(combinations)):
            if i - item["weight"] >= 0:
                combinations[i] = max(combinations[i], combinations[i-item["weight"]] + item["value"])
    return combinations[-1]

# print(knap2(5, [
#     {"weight": 5, "value": 60},
#     {"weight": 3, "value": 50},
#     {"weight": 4, "value": 70},
#     {"weight": 2, "value": 30}
# ]))

def quicksort(array=[12,4,5,6,7,3,1,15]):
    """Sort the array by using quicksort."""

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        # Don't forget to return something!
        return quicksort(less)+equal+quicksort(greater)  # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to handle the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array
#print(quicksort())


def search_rotated_array(A, t):
    def binary(A, t, l, r):
        if l <= r:
            m = l + (r-l) // 2
            if A[m] == t:
                return m
            if A[m] > A[l]:
                if t < A[m] and t >= A[l]:
                    return binary(A, t, l, m - 1)
                else:
                    return binary(A, t, m+1, r)
            else:
                if t <= A[r] and t > A[m]:
                    return binary(A, t, m+1, r)
                else:
                    return binary(A, t, l, m - 1)
        else:
            return -1
    return binary(A, t, 0, len(A) - 1)

#print(search_rotated_array([3,1], 1))

def search_no_size(reader, target):
    """
    :type reader: ArrayReader
    :type target: int
    :rtype: int
    """
    a = 2
    while reader.get(a) != 2147483647:
        if reader.get(a) == target:
            return a
        elif reader.get(a) > target:
            break
        else:
            a *= 2

    def binary(reader, t, l, r):
        if l <= r:
            m = l + (r - l) // 2
            item = reader.get(m)
            if item == t:
                return m
            elif item == -1 or item > t:
                return binary(reader, target, l, m - 1)
            elif item < t:
                return binary(reader, target, m + 1, r)
        else:
            return -1

    return binary(reader, target, a // 2, a)
class Readera:
    data = [-1,0,3,5,9,12]
    def get(self, d):
        if d > len(self.data):
            return 2147483647
        if d not in self.data:
            return -1
        else:
            return self.data.index(d)


# reader = Readera()
# print(search_no_size(reader,9))

def sparse_search(a, t):
    def binary(a, t, l, r):
        if l <= r:
            m = l + (r - l) // 2
            item = a[m]
            if item == "":
                ml = m -1
                mr = m +1
                if a[ml] < 0 or a[mr] > len(a) - 1:
                    return -1
                if a[ml] != "":
                    return binary(a, t, ml, m - 1)

            if item == t:
                return m
            elif item > t:
                return binary(a, t, l, m - 1)
            else:
                return binary(a, t, m + 1, r)
        else:
            return -1

    return binary(a, t, 0, len(a) - 1)

#print(sparse_search(["", "at", "", "", "ball"], "ball"))

def subset_sum(nums):
    stack = [(0, 0)]
    visited = {}
    fullSum = sum(nums)

    if fullSum % 2:
        return False

    while len(stack):
        index, sumNumbers = stack.pop()

        if index >= len(nums) or 2 * sumNumbers > fullSum:
            continue
        elif 2 * sumNumbers == fullSum:
            return True

        if (index, sumNumbers) not in visited:
            visited[(index, sumNumbers)] = True
        else:
            continue

        stack.append((index + 1, sumNumbers))
        stack.append((index + 1, sumNumbers + nums[index]))

    return False

#print(subset_sum(( [1, 5, 11, 5])))

def numIslands(grid: List[List[str]]) -> int:
    visited = set()
    islands = 0

    def dfs(r, c):
        if r < 0 or c < 0:
            return
        if r >= len(grid) or c >= len(grid[0]):
            return
        if (r, c) not in visited:
            visited.add((r, c))
            if grid[r][c] == "1":
                dfs(r + 1, c)
                dfs(r, c + 1)
                dfs(r - 1, c)
                dfs(r, c - 1)

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if (r,c) not in visited and grid[r][c] == "1":
                islands += 1
                dfs(r, c)
    return islands

#print(numIslands([["1","1","1"],["0","1","0"],["1","1","1"]]))


def numIslands2(m: int, n: int, positions: List[List[int]]) -> List[int]:
    visited = set()
    result = []

    def dfs(r, c):
        if r < 0 or c < 0:
            return
        if r >= m or c >= n:
            return
        if (r, c) not in visited:
            visited.add((r, c))
            if grid[r][c] == "1":
                dfs(r + 1, c)
                dfs(r, c + 1)
                dfs(r - 1, c)
                dfs(r, c - 1)

    grid = [[0 for x in range(n)] for x in range(m)]
    for r, c in positions:
        if r < m and c < n:
            grid[r][c] = "1"
        islands = 0
        visited = set()
        for r in range(m):
            for c in range(n):
                if (r, c) not in visited and grid[r][c] == "1":
                    islands += 1
                    dfs(r, c)
        result.append(islands)
    return result

#print(numIslands2(1,2, [[0,1],[0,0]]))

def isMatch( text, pattern):
    dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]

    dp[-1][-1] = True
    for i in range(len(text), -1, -1):
        for j in range(len(pattern) - 1, -1, -1):
            first_match = i < len(text) and pattern[j] in {text[i], '.'}
            if j+1 < len(pattern) and pattern[j+1] == '*':
                dp[i][j] = dp[i][j+2] or first_match and dp[i+1][j]
            else:
                dp[i][j] = first_match and dp[i+1][j+1]

    return dp[0][0]
# print(isMatch("mississippi","mis*is*p*."))

def videoStitching(clips: List[List[int]], T: int) -> int:
    result = 0
    temp = 0
    while temp < T:
        candidates = []
        for item in clips:
            if item[0] <= temp:
                if item[1] > temp:
                    candidates.append(item[1])
        if not len(candidates) and temp < T:
            return -1
        else:
            temp = max(candidates)
        result += 1

    return result

#print(videoStitching([[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], 10))

def add_binary_nums(x, y):
    max_len = max(len(x), len(y))

    x = x.zfill(max_len)
    y = y.zfill(max_len)

    # initialize the result
    result = ''

    # initialize the carry
    carry = 0

    # Traverse the string
    for i in range(max_len - 1, -1, -1):
        r = carry
        r += 1 if x[i] == '1' else 0
        r += 1 if y[i] == '1' else 0
        result = ('1' if r % 2 == 1 else '0') + result
        carry = 0 if r < 2 else 1  # Compute the carry.

    if carry != 0: result = '1' + result

    return result.zfill(max_len)

#print(add_binary_nums('1101', '100'))

def lcs(a,b,result):
    if not a or not b:
        return result
    A = a[-1]
    B = b[-1]
    if A==B:
        return lcs(a[:-1], b[:-1], result+1)
    else:
        return max(lcs(a[:-1], b, result), lcs(a, b[:-1], result))

#print(lcs("saab", "sazb", 0))

def number_to_bin(num):
    if num>1:
        number_to_bin(num//2)
    print(num%2)
#number_to_bin(16)

def multiply(n, m):
    ans = 0
    count = 0
    while (m):
        # check for set bit and left
        # shift n, count times
        if (m % 2 == 1):
            ans += n << count

            # increment of place value (count)
        count += 1
        m = int(m / 2)

    return ans
#print(multiply(3,4))
def longestOnes(A: List[int], K: int) -> int:
    j = 0
    for i in range(len(A)):
        if A[i] != 1:
            K -= 1
        if K < 0:
            if A[j] == 0:
                K += 1
            j += 1
    return i - j + 1
#print(longestOnes([1,1,1,0,0,0,1,1,1,1,0],2))

def task_scheduler(tasks, n):
    map = {}
    for task in tasks:
        if task in map:
            map[task] += 1
        else:
            map[task] = 1

    priority_queue = list(map.values())
    result = 0
    count = 0
    while count < len(tasks):
        biggest = priority_queue.pop()


def leastInterval(tasks, n):
    if n == 0:
        return len(tasks)
    import collections
    map = collections.defaultdict(int)
    for task in tasks:
        map[task] += 1
    s = sorted(list(map.values()), reverse=True)
    gaps = s[0] - 1
    slots = gaps * n
    for i in range(1, len(s)):
        slots -= min(s[i], gaps)

    return max(slots + len(tasks), len(tasks))

#print(leastInterval(["A","A","A","B","B","B"],2))
import collections

def criticalConnections2(connections: List[List[int]], n) -> List[List[int]]:

    graph = collections.defaultdict(list)
    unique = set()
    for c in connections:
        graph[c[0]].append(c[1])
        unique.add(c[0])
        unique.add(c[1])

    def dfs(edges):
        stack = [edges[0][0]]
        graph = collections.defaultdict(list)
        for c in edges:
            graph[c[0]].append(c[1])
            graph[c[1]].append(c[0])
        visited = set()
        while stack:
            item = stack.pop()
            visited.add(item)
            for x in graph[item]:
                if x not in visited:
                    stack.append(x)

        return visited

    final = []
    temp = [x for x in connections]
    n = len(temp)
    for i in range(n):
        item = temp.pop(i)
        result = dfs(temp)
        if result != unique:
            final.append(item)
        temp.insert(i, item)
    return final

#print(criticalConnections2([[0,1],[1,2],[2,0],[1,3]], 4))
def a():
    result = set()
    def dfs(node, graph):
        nonlocal result
        if node != None:
            result.add(node)
            for x in graph[node]:
                if x not in result:
                    dfs(x, graph)
    da = [[0,1],[1,2],[2,0],[1,3]]
    graph = collections.defaultdict(list)
    for c in da:
        graph[c[0]].append(c[1])
    dfs(0, graph)
    return result
#print(a())

seen = set()
def isHappy(n: int) -> bool:
    if n == 1:
        return True
    elif n in seen:
        return False
    seen.add(n)
    s = str(n)
    res = 0
    for x in s:
        res += int(x)**2
    res = int(res)
    return isHappy(res)
print(isHappy(10))
# anagrams and palindrom

def shortestToChar(S: str, C: str) -> List[int]:
    result = []
    for i in range(len(S)):
        if S[i] == C:
            result.append(0)
        else:
            m = i - 1
            n = i + 1
            temp = False
            while m >= 0:
                if S[m] == C:
                    temp = abs(i - m)
                    break
                m -= 1
            while n < len(S):
                if S[n] == C:
                    if not temp:
                        temp = abs(i - n)
                    else:
                        temp = min(temp, abs(i - n))
                    break
                n += 1
            result.append(temp)
    return result

def shortestToChar2(S: str, C: str) -> List[int]:
    result = []
    left = float("Inf")
    right = S.index(C)
    for i, c in enumerate(S):
        if c == C:
            result.append(0)
            left = i
            try:
                right = S[i + 1:].index(C) + i
            except:
                right = float("Inf")
        else:
            result.append(min(abs(i - left), abs(i - right)))

    return result

#print(shortestToChar2("loveleetcode", "e"))

def shortestSubarray(A: List[int], K: int) -> int:
    i = 0
    j = 0
    result = float("Inf")
    while i < len(A) and j < len(A):
        if sum(A[i:j]) == K:
            result = min(abs(i - j), result)
            i += 1
        elif sum(A[i:j]) < K:
            j += 1
        elif sum(A[i:j]) > K:
            i += 1

    for x in range(i, len(A)):
        if sum(A[i:]) == K:
            result = min(abs(i - len(A)), result)

    return -1 if result == float("Inf") else result

#print(shortestSubarray([48,99,37,4,-31],140))
# anagrams and palindrom

def productExceptSelf(nums: List[int]) -> List[int]:
    result = [1 for x in nums]
    left = 1
    right = 1
    for i, n in enumerate(nums):
        result[i] *= left
        left *= n
        result[~i] *= right
        right *= nums[~i]
    return result

#print(productExceptSelf([2,3,4,5]))

def loopbothways(n):
    i = 0
    while i < len(n) // 2:
        print(n[i])
        print(n[~i])
        i += 1

#print(loopbothways([1,2,3,4,5,6]))

def isValidSudoku(board: List[List[str]]) -> bool:
    row = 0
    while row < 9:
        seen = set()
        for n in board[row]:
            if n in seen and n != ".":
                return False
            seen.add(n)
        row += 1

    col = 0
    while col < 9:
        seen = set()
        for n in range(9):
            item = board[n][col]
            if item in seen and item != ".":
                return False
            seen.add(item)
        col += 1

    row = 0
    col = 0
    for _ in range(3):
        for _ in range(3):
            seen = set()
            for i in range(3):
                for j in range(3):
                    v = board[row + i][col + j]
                    if v in seen and v != ".":
                        return False
                    seen.add(v)
            col += 3
            if col == 9:
                col = 0
        row += 3

    return True

#print(isValidSudoku([["7","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]))
