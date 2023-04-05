import homework3_tvm5513
import unittest
import time

start = time.process_time()

class Testhw3(unittest.TestCase):

    def test_get_board(self):
        p = homework3_tvm5513.TilePuzzle([[1, 2], [3, 0]])
        self.assertEqual(p.get_board(), [[1, 2], [3, 0]])

    def test_create_tile_puzzle(self):
        p = homework3_tvm5513.create_tile_puzzle(3, 3)
        self.assertEqual(p.get_board(), [[1, 2, 3], [4, 5, 6], [7, 8, 0]])

    def test_perform_move(self):
        p = homework3_tvm5513.create_tile_puzzle(3, 3)
        self.assertEqual(p.perform_move("up"), True)
        print(p.get_board())

    def test_is_solved_and_scramble(self):
        p = homework3_tvm5513.TilePuzzle([[1, 2], [3, 0]])
        self.assertEqual(p.is_solved(), True)

    def test_copy(self):
        p = homework3_tvm5513.create_tile_puzzle(3, 3)
        p2 = p.copy()
        self.assertEqual(p.get_board() == p2.get_board(), True)

    def test_successors(self):
        p = homework3_tvm5513.create_tile_puzzle(3, 3)
        for m, n in p.successors():
            print(m, n.get_board())

    def test_find_solution_iddfs(self):
        b = [[4, 1, 2], [0, 5, 3], [7, 8, 6]]
        p = homework3_tvm5513.TilePuzzle(b)
        solutions = p.find_solutions_iddfs()
        self.assertEqual(next(solutions), ['up', 'right', 'right', 'down', 'down'])

    def test_find_solution_a_star(self):
        b = [[4, 1, 2], [0, 5, 3], [7, 8, 6]]
        p = homework3_tvm5513.TilePuzzle(b)
        self.assertEqual(p.find_solution_a_star(), ['up', 'right', 'right', 'down', 'down'])

    def test_find_path(self):
        scene = [[False, False, False], [False, True, False], [False, False, False]]
        self.assertEqual(homework3_tvm5513.find_path((0, 0), (2, 1), scene), [(0, 0), (1, 0), (2, 1)])

    def test_get_board_dominoes(self):
        b = [[False, False], [False, False]]
        g = homework3_tvm5513.DominoesGame(b)
        self.assertEqual(g.get_board(), [[False, False], [False, False]])

    def test_create_dominoes_game(self):
        p = homework3_tvm5513.create_dominoes_game(2, 2)
        self.assertEqual(p.get_board(), [[False, False], [False, False]])

    def test_reset(self):
        b = [[True, False], [True, False]]
        g = homework3_tvm5513.DominoesGame(b)
        g.reset()
        self.assertEqual(g.get_board(), [[False, False], [False, False]])

    def test_is_legal_move(self):
        b = [[False, False], [False, False]]
        g = homework3_tvm5513.DominoesGame(b)
        self.assertEqual(g.is_legal_move(0, 0, True), True)
        self.assertEqual(g.is_legal_move(0, 0, False), True)

    def test_legal_moves(self):
        g = homework3_tvm5513.create_dominoes_game(3, 3)
        self.assertEqual(list(g.legal_moves(True)), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        self.assertEqual(list(g.legal_moves(False)), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def test_perform_move_dominoes(self):
        g = homework3_tvm5513.create_dominoes_game(3, 3)
        g.perform_move(0, 1, True)
        print(g.get_board())

    def test_game_over(self):
        b = [[False, False], [False, False]]
        g = homework3_tvm5513.DominoesGame(b)
        self.assertEqual(g.game_over(True), False)
        self.assertEqual(g.game_over(False), False)

    def test_copy_dominoes(self):
        g = homework3_tvm5513.create_dominoes_game(4, 4)
        g2 = g.copy()
        self.assertEqual(g.get_board() == g2.get_board(), True)

    def test_successors_dominoes(self):
        b = [[False, False], [False, False]]
        g = homework3_tvm5513.DominoesGame(b)
        for m, n in g.successors(True):
            print(m, n.get_board())

    def test_get_best_move(self):
        b = [[False] * 3 for i in range(3)]
        g = homework3_tvm5513.DominoesGame(b)
        self.assertEqual(g.get_best_move(True, 1), ((0, 1), 2, 6))
        self.assertEqual(g.get_best_move(True, 2), ((0, 1), 3, 10))

print(time.process_time() - start)

if __name__ == '__main__':
    unittest.main()

