############################################################
# CMPSC 442: Homework 3
############################################################

student_name = "Trisha Mandal"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random
import copy
import math
from queue import PriorityQueue


############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    temp = 0
    board = []
    dimension = rows * cols
    for a in range(0, rows):
        board.append([])
        for b in range(0, cols):
            temp = temp + 1
            if temp != dimension:
                board[a].append(temp)
            if temp == dimension:
                board[a].append(0)

    return TilePuzzle(board)


class TilePuzzle(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.columns = len(board[0])

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        tboard = self.get_board()
        points = None

        for a in range(0, self.rows):
            for b in range(0, self.columns):
                temp = tboard[a][b]
                if temp == 0:
                    points = tuple([a, b])
                    break

        oldvalue, newvalue = None, None

        if direction == "right" and points[1] != len(tboard) - 1:
            newvalue = tuple([points[0], points[1] + 1])
        elif direction == "right" and points[1] == len(tboard) - 1:
            return False

        if direction == "up" and points[0] != 0:
            newvalue = tuple([points[0] - 1, points[1]])
        elif direction == "up" and points[0] == 0:
            return False

        if direction == "left" and points[1] != 0:
            newvalue = tuple([points[0], points[1] - 1])
        elif direction == "left" and points[1] == 0:
            return False

        if direction == "down" and points[0] != len(tboard) - 1:
            newvalue = tuple([points[0] + 1, points[1]])
        elif direction == "down" and points[0] == len(tboard) - 1:
            return False

        for a in range(0, self.rows):
            for b in range(0, self.columns):
                tup1 = tuple([a, b])
                if tup1 == newvalue:
                    oldvalue = tboard[a][b]
                    tboard[a][b] = 0

        for a in range(0, self.rows):
            for b in range(0, self.columns):
                tup2 = tuple([a, b])
                if tup2 == points:
                    tboard[a][b] = oldvalue

        self.board = tboard

        return True

    def scramble(self, num_moves):
        for a in range(0, num_moves):
            ran = random.choice(["up", "down", "left", "right"])
            self.perform_move(ran)

    def is_solved(self):
        tboard = self.get_board()
        puzz = create_tile_puzzle(self.rows, self.columns)
        firstboard = puzz.get_board()
        if tboard != firstboard:
            return False
        if tboard == firstboard:
            return True

    def copy(self):
        cop = []
        for a in self.board:
            cop.append(a[:])
        solution = TilePuzzle(cop)
        return solution

    def successors(self):
        plays = ["left", "right", "up", "down"]

        for a in range(0, len(plays)):
            p2 = self.copy()
            r = p2.perform_move(plays[a])
            if r:
                yield tuple([plays[a], p2])

    def iddfs_helper(self, limit, moves, new):
        if new is None:
            new = self.copy()
        if moves is None:
            moves = []
        if new.is_solved():
            yield moves
        lm = len(moves)
        if limit > lm:
            for addingm, nplay in new.successors():
                nmoves = copy.copy(moves)
                nmoves.append(addingm)
                solution = self.iddfs_helper(limit, nmoves, nplay)
                yield from solution

    def find_solutions_iddfs(self):
        first = 0
        while True:
            c = list(self.iddfs_helper(first, None, None))
            if c:
                for a in range(0, len(c)):
                    yield c[a]
                break
            first = first + 1

    def Manhattandistance(self, goalboard, currentboard):
        mandist = 0
        dic = {}
        board = goalboard
        for i in range(0, self.rows):
            for j in range(0, self.columns):
                tempboard = board[i][j]
                tupletemp = tuple([i, j])
                dic[tempboard] = tupletemp

        for i in range(0, self.rows):
            for j in range(0, self.columns):
                newtup = ([i, j])
                oldtup = dic[currentboard[i][j]]
                d1 = abs(newtup[1] - oldtup[1])
                d2 = abs(newtup[0] - oldtup[0])
                mandist = mandist + d1 + d2
        return mandist

    def find_solution_a_star(self):

        pqueue = PriorityQueue()
        way = [(None, -1)]
        goalb = create_tile_puzzle(len(self.get_board()), len(self.get_board()[0])).get_board()
        if goalb == self.get_board():
            return []
        marker = 0
        pqueue.put(tuple([-1, tuple([self.get_board(), None, 0, -1])]))

        same = set()
        fpath = []
        while pqueue:
            nnode = pqueue.get()
            cost = nnode[1][2]
            tempnew = nnode[1][0]
            node13 = nnode[1][3]
            node11 = nnode[1][1]
            new = TilePuzzle(tempnew)
            if new.is_solved():
                cNode = way[node13]
                fpath.append(node11)

                if cNode[1] == -1:
                    break

                while True:
                    fpath.append(cNode[0])
                    if cNode[1] == 0:
                        break
                    if cNode[1] != 0:
                        cNode = way[cNode[1]]
                break

            for m, nm in new.successors():
                listtup = [tuple(nm.get_board()[a]) for a in range(0, len(self.get_board()))]
                if tuple(listtup) not in same:
                    dis = self.Manhattandistance(goalb, nm.get_board())
                    tup1 = tuple([node11, node13])
                    pqueue.put(tuple([cost + dis, tuple([nm.get_board(), m, cost + 1, way.index(tup1)])]))
                    if way:
                        ind =way.index(tup1)
                        tup2 = tuple([m, ind])
                        way.append(tup2)
                    if not way:
                        tup3 = tuple([m, None])
                        way.append(tup3)
                    final = tuple(listtup)
                    same.add(final)
            marker = marker + 1
        fpath.reverse()
        solution = fpath
        return solution


############################################################
# Section 2: Grid Navigation
############################################################


def euclidean(p1, p2):
    solution = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return solution


def perform_move(direction, current_point, scene):
    r = len(scene)
    c = len(scene[0])
    pointx = current_point[1]
    pointy = current_point[0]
    sqroot = math.sqrt(2)
    if direction == 'up':
        if not scene[pointy - 1][pointx]:
            if pointy > 0:
                return (pointy - 1, pointx), 1
        return False
    if direction == 'down':
        if not scene[pointy + 1][pointx]:
            if pointy < r - 1:
                return (pointy + 1, pointx), 1
        return False
    if direction == 'left':
        if not scene[pointy][pointx - 1]:
            if pointx > 0:
                return (pointy, pointx - 1), 1
        return False
    if direction == 'right':
        if not scene[pointy][pointx + 1]:
            if pointx < c - 1:
                return (pointy, pointx + 1), 1
        return False
    if direction == 'upleft':
        if not scene[pointy - 1][pointx - 1]:
            if pointx > 0:
                if pointy > 0:
                    return (pointy - 1, pointx - 1), sqroot
        return False
    if direction == 'upright':
        if pointx < c - 1:
            if not scene[pointy - 1][pointx + 1]:
                if pointy > 0:
                    return (pointy - 1, pointx + 1), sqroot
        return False
    if direction == 'downleft':
        if not scene[pointy + 1][pointx - 1]:
            if pointx > 0:
                if pointy < r - 1:
                    return (pointy + 1, pointx - 1), sqroot
        return False
    if direction == 'downright':
        if pointx < c - 1:
            if not scene[pointy + 1][pointx + 1]:
                if pointy < r - 1:
                    return (pointy + 1, pointx + 1), sqroot
        return False
    else:
        return False


def successor(current_point, scene):
    moves = ['upleft', 'upright', 'downleft', 'downright', 'up', 'down', 'left', 'right']
    for play in moves:
        if perform_move(play, current_point, scene):
            yield perform_move(play, current_point, scene)


def find_path(start, goal, scene):
    scene1 = scene[goal[0]][goal[1]]
    scene2 = scene[start[0]][start[1]]
    if scene1 or scene2 or start == goal:
        return None
    explored = set()
    explored.add(start)
    pq = PriorityQueue()
    euc = euclidean(start, goal)
    pq.put((euc, euc, 0, [start]))
    if pq.empty():
        return None
    else:
        while pq:
            total, e, t, p = pq.get()
            if e == 0:
                return p
            else:
                l = len(p) - 1
                explored.add(p[l])
                for npoint, ed in successor(p[l], scene):
                    if npoint in explored:
                        break
                    if npoint not in explored:
                        e = euclidean(npoint, goal)
                        total = e + t + ed
                        pq.put((total, e, t + ed, p + [npoint]))
    return None


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def solve_distinct_disks(length, n):
    pass


############################################################
# Section 4: Dominoes Game
############################################################


def create_dominoes_game(rows, cols):
    new_dominoes = [[False for a in range(0,cols)] for b in range(0,rows)]
    solution = DominoesGame(new_dominoes)
    return solution


class DominoesGame(object):

    def __init__(self, board):
        self.board = board
        self.columns = len(board[0])
        self.rows = len(board)

    def get_board(self):
        return self.board

    def reset(self):
        self.board = [[False for a in range(0,len(self.board[0]))] for b in range(0, len(self.board))]

    def is_legal_move(self, row, col, vertical):
        if not vertical:
            if 0 <= row < len(self.board):
                if 0 <= col + 1 < len(self.board[0]):
                    if not self.board[row][col + 1]:
                        if not self.board[row][col]:
                            return True
        if vertical:
            if 0 <= row + 1 < len(self.board):
                if 0 <= col < len(self.board[0]):
                    if not self.board[row + 1][col]:
                        if not self.board[row][col]:
                            return True
        return False

    def legal_moves(self, vertical):
        for a in range(0, len(self.board)):
            for b in range(0, len(self.board[0])):
                if self.is_legal_move(a, b, vertical):
                    yield a, b

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            if vertical:
                self.board[row + 1][col], self.board[row][col] = True, True
            if not vertical:
                self.board[row][col], self.board[row][col + 1] = True, True

    def eval(self, vertical):
        solution = [len(list(self.legal_moves(vertical))), len(list(self.legal_moves(not vertical)))]
        return solution

    def game_over(self, vertical):
        b = list(self.legal_moves(vertical))
        if not b:
            return True
        return False

    def copy(self):
        return copy.deepcopy(self)

    def successors(self, vertical):
        for play in self.legal_moves(vertical):
            cop = self.copy()
            cop.perform_move(play[0], play[1], vertical)
            yield play, cop

    def get_random_move(self, vertical):
        solution = random.choice(list(self.legal_moves(vertical)))
        return solution

    def maximum(self, node, vertical, limit, alpha, beta):
        if self.game_over(vertical) or limit == 0:
            numb = self.eval(vertical)
            return node, numb[0] - numb[1], 1
        else:
            marker = 0
            v = - math.inf
            for play, nnode in self.successors(vertical):
                t1, t2, t3 = nnode.minimum(play, not vertical, limit - 1, alpha, beta)
                # nmove = t1
                nscore = t2
                ncount = t3
                marker = marker + ncount
                if v < nscore:
                    node = play
                    v = nscore
                if beta <= v:
                    return node, v, marker
                else:
                    maxi = max(v, alpha)
                    alpha = maxi
            return node, v, marker

    def minimum(self, node, vertical, limit, alpha, beta):

        if self.game_over(vertical) or limit == 0:
            num = self.eval(vertical)
            return node, num[1] - num[0], 1
        else:
            marker = 0
            v = math.inf
            for play, nnode in self.successors(vertical):
                t1, t2, t3 = nnode.maximum(play, not vertical, limit - 1, alpha, beta)
                # nmove = t1
                nscore = t2
                ncount = t3
                marker = marker + ncount
                if v > nscore:
                    node = play
                    v = nscore
                if alpha >= v:
                    return node, v, marker
                else:
                    mini = min(v, beta)
                    beta = mini
            return node, v, marker

    def get_best_move(self, vertical, limit):
        solution = self.maximum(None, vertical, limit, -math.inf, math.inf)
        return solution


############################################################
# Section 5: Feedback
############################################################

feedback_question_1 = """
I spent around 30-35 hours on this code.
"""

feedback_question_2 = """
I could not figure out how to do linear disk. 
It was just too tough for me. 
"""

feedback_question_3 = """
Trying out the GUI and stuff was super cool in this project.
"""
