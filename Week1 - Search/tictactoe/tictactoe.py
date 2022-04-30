"""
Tic Tac Toe Player
"""

import math
from operator import truediv
from pickle import FALSE
from queue import Empty

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    xcount = 0
    ocount = 0

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == X:
                xcount += 1
            if board[i][j] == O:
                ocount += 1

    if xcount == 0 and ocount == 0:
        return X

    if xcount == ocount:
        return X

    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    moves = set()

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                moves.add((i,j))

    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    #making deep copy of board
    newboard = [row[:] for row in board]

    i = action[0]
    j = action[1]

    if newboard[i][j] != EMPTY:
        raise Exception
    else:
        newboard[i][j] = player(newboard)

    return newboard

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(len(board)):
        x = all(ele == X for ele in board[i])
        o = all(ele == O for ele in board[i])

        if x:
            return X
        if o:
            return O

    for i in range(len(board)):
        
        x = True
        o = True
        for j in range(len(board[i])):
            if board[j][i] == EMPTY:
                o = False
                x = False
            if board[j][i] == X:
                o = False
            else:
                x = False

        if x:
            return X
        if o:
            return O

    x = True
    o = True
    for i in range(len(board)):
        if board[i][i] == EMPTY:
            o = False
            x = False
        if board[i][i] == X:
            o = False
        else:
            x = False

    
    if x:
        return X
    if o:
        return O


    x = True
    o = True
    for i in range(len(board)):
        if board[i][len(board) - 1 - i] == EMPTY:
            o = False
            x = False
        if board[i][len(board) - 1 - i] == X:
            o = False
        else:
            x = False

    
    if x:
        return X
    if o:
        return O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board):
        return True

    filled = True

    for i in range(len(board)):
        filled = filled and all(ele != EMPTY for ele in board[i])

    if filled:
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    w = winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    
    return 0


def maxValue(board):
    
    if(terminal(board)):
        return utility(board),()

    v = -100
    act = ()

    for action in actions(board):
        val = minValue(result(board, action))
        
        if val[0] > v:
            act = action
            v = val[0]

    return v,act

def minValue(board):
    
    if(terminal(board)):
        return utility(board),()

    v = 100
    act = ()

    for action in actions(board):
        val = maxValue(result(board, action))
        
        if val[0] < v:
            act = action
            v = val[0]

    return v,act


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None

    p = player(board)

    if p == X:
        return maxValue(board)[1]
    else:
        return minValue(board)[1]
