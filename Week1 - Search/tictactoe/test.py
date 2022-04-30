from queue import Empty
import tictactoe as t


X = "X"
O = "O"
chance = X
EMPTY = None

board = [[X, O, EMPTY],
            [O, O, X],
            [X, X, O]]

print(t.terminal(board))