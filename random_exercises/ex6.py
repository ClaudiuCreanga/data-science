from typing import List, Dict
from collections import defaultdict

def bomb_baby(x, y):
    from heapq import heappop, heappush, heapify
    q = [(int(x) - 1, 1, 1, 2), (int(y) - 1, 1, 2, 1)]
    heapify(q)
    seen = set()
    while q:
        cost, generations, m, f = heappop(q)
        if str(m) == x and str(f) == y:
            return str(generations)
        if m > int(x) or f > int(y):
            continue
        seen.add((m, f))
        m_diff = int(x) - m
        f_diff = int(y) - f
        if (m+f, f) not in seen:
            heappush(q, (generations + f_diff , generations + 1, m+f, f))
        if (m, f+m) not in seen:
            heappush(q, (generations + m_diff, generations + 1, m, f+m))

    return "impossible"
#
# print(bomb_baby("2", "1"))
# print(bomb_baby("2", "4"))
# print(bomb_baby("4", "7"))

def answer(x, y):
    generations = 0
    m, f = int(x), int(y)
    while True:
        if m == 1 and f == 1:
            return str(generations)
        elif m < 1 or f < 1 or m == f:
            return "impossible"
        else:
            if m > f:
                if m > 100 * f:
                    generations += int(m / f)
                    m = m - (int(m / f) * f)
                else:
                    m -= f
                    generations += 1
            else:
                if f > 100 * m:
                    generations += int(f / m)
                    f = f - (int(f / m) * m)
                else:
                    f -= m
                    generations += 1

# print(answer("2", "1"))
# print(answer("2", "4"))
# print(answer("4", "7"))

def fuel_pellets(n):

    def most_trailing_zeros(x):
        count = 0
        while ((x & 1) == 0):
            x = x >> 1
            count += 1

        return count

    N = int(n)
    count = 0
    while N >= 1:
        if N == 1:
            return count
        else:
            if N % 2 != 0:
                plus = N + 1
                minus = N - 1
                up = most_trailing_zeros(plus)
                down = most_trailing_zeros(minus)
                if up > down and (plus // 2 < minus):  # when going up is actually beneficial not like in case of 3
                    N += 1
                else:
                    N -= 1
            else:
                N //= 2
            count += 1

    return count

#
# print(fuel_pellets(1))
# print(fuel_pellets(2))
# print(fuel_pellets(3))
# print(fuel_pellets(4))
# print(fuel_pellets(15))
# print(fuel_pellets(25))

def solution_escape(map):
    # do with nodes
    from collections import deque
    n_r = len(map)
    n_c = len(map[0])
    goal = (n_r - 1, n_c - 1)
    start = (0, 0, False, 1)
    seen = set()
    q = deque()
    q.append(start)

    def map_insert(q, r, c, d, i, seen):
        if (r, c, d) not in seen:
            if map[r][c] == 1:
                if not d:
                    q.append((r, c, True, i))
            else:
                q.append((r, c, False, i))


    while q:
        r, c, d, i = q.popleft()
        if (r, c) == goal:
            return i
        seen.add((r, c, d))
        i += 1
        if r + 1 < n_r:
            map_insert(q, r + 1, c, d, i, seen)
        if c + 1 < n_c:
            map_insert(q, r, c + 1, d, i, seen)
        if r - 1 >= 0:
            map_insert(q, r - 1, c, d, i, seen)
        if c - 1 >= 0:
            map_insert(q, r, c - 1, d, i, seen)

class Node:

    def __init__(self, p, e, c):
        self.p = p
        self.e = e
        self.c = c

def solution_escape2(map):
    from collections import deque
    start = Node(1, None, (0, 0))
    n_r = len(map)
    n_c = len(map[0])
    goal = Node(None, 1, (n_r - 1, n_c - 1))
    q = deque()
    q.append(start)
    seen = set()
    for r in range(n_r):
        for c in range(n_c):
            if map[r][c] == 0:
                map[r][c] = Node(float("Inf"), float("Inf"), (r, c))
            else:
                map[r][c] = None

    def is_valid_pos(r, c):
        if 0 <= r < n_r and 0 <= c < n_c and map[r][c]:
            return True
        return False

    def valid_ckeck(q, p, e, r, c, seen):
        if is_valid_pos(r + 1, c):
            map_insert(q, p, e, r + 1, c, seen)
        if is_valid_pos(r, c + 1):
            map_insert(q, p, e, r, c + 1, seen)
        if is_valid_pos(r - 1, c):
            map_insert(q, p, e, r - 1, c, seen)
        if is_valid_pos(r, c - 1):
            map_insert(q, p, e, r, c - 1, seen)

    def map_insert(q, p, e, r, c, seen):
        if (r, c) not in seen and map[r][c]:
            q.append(Node(p, e, (r, c)))

    while q:
        node = q.popleft()
        if node.c not in seen:
            r, c = node.c
            if map[r][c].p > node.p:
                map[r][c].p = node.p
            if node.c == goal.c:
                break
            p = node.p
            valid_ckeck(q, p+1, node.e, r, c, seen)

    q = deque()
    q.append(goal)
    seen = set()
    while q:
        node = q.popleft()
        if node.c not in seen:
            r, c = node.c
            if map[r][c].e > node.e:
                map[r][c].e = node.e
            if node.c == start.c:
                break
            e = node.e
            valid_ckeck(q, node.p,  e+1, r, c, seen)

    answer = float("Inf")
    for r in range(n_r):
        for c in range(n_c):
            if map[r][c] == None:
                options = []
                if is_valid_pos(r-1, c):
                    top = map[r-1][c].p
                    options.append(top)
                if is_valid_pos(r+1, c):
                    bottom = map[r+1][c].e
                    options.append(bottom)
                if is_valid_pos(r, c-1):
                    left = map[r][c-1].p
                    options.append(left)
                if is_valid_pos(r, c+1):
                    right = map[r][c+1].e
                    options.append(right)
                while len(options) > 1:
                    v = options.pop(0)
                    for j in range(len(options)):
                        answer = min(answer, v + options[j] + 1)
    return answer

def solution_escape3(map):
    from copy import deepcopy
    from heapq import heappush, heappop
    n_r = len(map)
    n_c = len(map[0])
    goal_r = n_r -1
    goal_c = n_c - 1
    def is_valid_pos(r, c, seen, newgrid):
        if 0 <= r < n_r and 0 <= c < n_c and (r,c) not in seen and newgrid[r][c] == 0:
            return True
        return False

    def shortest_path_a_star(newgrid):
        start = (0, 1, 0, 0)
        goal = (goal_r, goal_c)
        q = [start]
        seen = set()
        while q:
            cost, w, r, c = heappop(q)
            if (r, c) == goal:
                return w
            seen.add((r, c))
            if is_valid_pos(r + 1, c, seen, newgrid):
                m_d = abs(r+1-goal_r) + abs(c - goal_c) + w
                heappush(q, (m_d, w+1, r+1, c))
            if is_valid_pos(r, c + 1, seen, newgrid):
                m_d = abs(r-goal_r) + abs(c+1 - goal_c) + w
                heappush(q, (m_d, w+1, r, c+1))
            if is_valid_pos(r - 1, c, seen, newgrid):
                m_d = abs(r-1-goal_r) + abs(c - goal_c) + w
                heappush(q, (m_d, w+1, r-1, c))
            if is_valid_pos(r, c - 1, seen, newgrid):
                m_d = abs(r-goal_r) + abs(c-1 - goal_c) + w
                heappush(q, (m_d, w+1, r, c-1))
        return float("inf")


    ones = [(r,c) for r in range(n_r) for c in range(n_c) if map[r][c] == 1]
    answer = float('inf')
    if not ones:
        answer = n_c + n_r - 1
    for r,c in ones:
        newgrid = deepcopy(map)
        newgrid[r][c] = 0
        answer = min(answer, shortest_path_a_star(newgrid))
        if answer ==  n_c + n_r - 1:
            return answer
    return answer



print(solution_escape3([
   [0, 1, 0, 1, 0, 0, 0],
   [0, 0, 0, 1, 0, 1, 0]
]
)) # 10

print(solution_escape3([
                       [0, 1, 1, 0],
                       [0, 0, 0, 1],
                       [1, 1, 0, 0],
                       [1, 1, 1, 0]
])) # 7

print(solution_escape3([
                       [0, 0],
                       [0, 0],
])) # 3
print(solution_escape3([
                       [0, 0],
                       [0, 0],
                       [0, 0],
])) # 4

print(solution_escape3([
                       [0, 1, 1, 0],
                       [0, 0, 0, 1],
                       [1, 1, 0, 0],
])) # 6

print(solution_escape3([[0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0]])) # 11