import copy
import sys
import numpy as np
from scipy.signal import convolve2d
import torch


device='cuda'

# kernels
horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


class State:
    def __init__(self, parent,isroot,board,cp,prior_policy,arow=None,acul=None,Q=torch.tensor(0).to(device),N=torch.tensor(sys.float_info.epsilon).to(device)):
        self.__parent = parent
        if not isroot:
            self.__board=copy.deepcopy(parent.get_board())
        else:
            self.__board=copy.deepcopy(board)
        self.__Q=Q
        self.__N=N
        self.__children=[]
        self.__player=cp
        self.__arow=arow
        self.__acul=acul
        self.__ppolicy=prior_policy
        self.__action_policies=None
        self.__mcts_policy=None
        self.__value=None


    def get_Q(self):
        return self.__Q
    def get_N(self):
        return self.__N
    def set_Q(self,value):
        self.__Q=value
    def set_N(self,count):
        self.__N=count
    def get_children(self):
        return self.__children
    def change_mcts_policy(self,probs):
        self.__mcts_policy=probs
    def get_mcts_policy(self):
        return self.__mcts_policy
    def get_prior_policy(self):
        return self.__ppolicy
    def add_child(self,child):
        self.__children.append(child)
    def get_player(self):
        return self.__player
    def set_player(self,cp):
        self.__player=cp
    def get_board(self):
        return self.__board
    def change_board(self,row,cul):
        self.__board[row][cul]=self.__player
    def set_children(self,children):
        self.__children=children
    def get_arow(self):
        return self.__arow
    def undo_move(self,row,cul):
        self.__board[row][cul] = 0
    def get_acul(self):
        return self.__acul
    def get_parent(self):
        return self.__parent
    def set_inferences(self,val,probs):
        self.__value=val
        self.__action_policies=probs
    def get_inferences(self):
        return self.__value,self.__action_policies


    def legal_moves(self):
        moves = []
        for i in range(self.__board.shape[1]):
            blanks = np.where(self.__board[:, i] == 0)
            if len(blanks[0]) != 0:
                moves.append([blanks[0][-1], i])
        return moves

    def evaluation(self,cp=None):
        if cp is None:
            cp=self.__player
        for kernel in detection_kernels:
            result_board = convolve2d(self.__board, kernel, mode="valid")
            if 4 in result_board*cp.item():
                return "win"
            if -4 in result_board*cp.item():
                return "loss"

        if len(np.where(self.__board == 0)[0]) == 0:
            return "draw"

        return "is not terminal"

    def plot(self):
        print(str(self.__board).replace(' [', '').replace('[', '').replace(']', ''))
        print("player: "+str(self.__player))
        print("visit count: "+str(self.__N))
        print("value: "+str(self.__Q))
        print("action row: "+str(self.__arow))
        print("action row: "+str(self.__acul))
        print("*"*20)