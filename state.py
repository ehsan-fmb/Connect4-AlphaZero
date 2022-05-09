import copy
import sys
import numpy as np
from scipy.signal import convolve2d
import torch


# kernels
horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


class State:
    def __init__(self, parent,isroot,board,cp,prior_policy,arow=None,acul=None,Q=0,N=sys.float_info.epsilon):
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
        self.__final=False
        self.__mcts_policy=None
        self.__value=None
        self.__encoded_board=None


    def get_Q(self):
        return self.__Q
    def set_final(self):
        self.__final=True
    def get_final(self):
        return self.__final
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
        self.__board[row][cul]=-1*self.__player
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
    def get_encoded_board(self):
        return self.__encoded_board
    
    
    def encode_board(self):
        board=self.__board
        encoded = np.zeros([1,3,6,7]).astype(int)
        for row in range(6):
            for col in range(7):
                if board[row,col] ==1:
                    encoded[:,1,row, col] = 1
                if board[row,col] ==-1:
                    encoded[:,0,row, col] = 1
        
        if  self.__player== 1:
            encoded[:,2,:,:] = 1 # player to move
        self.__encoded_board=encoded


    def legal_moves(self):
        moves = []
        for i in range(self.__board.shape[1]):
            blanks = np.where(self.__board[:, i] == 0)
            if len(blanks[0]) != 0:
                moves.append([blanks[0][-1], i])
        return moves

    
    def evaluation1(self,cp=None):
        if cp is None:
            cp=-1*self.__player
        for kernel in detection_kernels:
            result_board = convolve2d(self.__board, kernel, mode="valid")
            if 4 in result_board*cp:
                return "win"
            if -4 in result_board*cp:
                return "loss"

        if len(np.where(self.__board == 0)[0]) == 0:
            return "draw"

        return "is not terminal"    
    
    def evaluation(self,cp=None):
        if cp is None:
            cp=self.__player
        for kernel in detection_kernels:
            result_board = convolve2d(self.__board, kernel, mode="valid")
            if 4 in result_board or -4 in result_board:
                return "terminal"

        if len(np.where(self.__board == 0)[0]) == 0:
            return "draw"

        return "is not terminal"

    def plot(self):
        print(str(self.__board).replace(' [', '').replace('[', '').replace(']', ''))
        # print(str(self.__encoded_board[0,0,:,:]))
        # print(str(self.__encoded_board[0,1,:,:]))
        # print(str(self.__encoded_board[0,2,:,:]))
        print("player: "+str(self.__player))
        print("visit count: "+str(self.__N))
        print("final status: "+str(self.__final))
        print("Q value: "+str(self.__Q))
        print("value inference: "+str(self.get_inferences()[0]))
        print("action probs: "+str(self.get_inferences()[1]))
        print("mcts probs:"+str(self.get_mcts_policy()))
        print("action row: "+str(self.__arow))
        print("action cul: "+str(self.__acul))
        print("ones: "+str(np.count_nonzero(self.__board==1)))
        print("minus ones: "+str(np.count_nonzero(self.__board==-1)))
        print("number of children: "+str(len(self.__children)))
        values=[]
        for i in self.__children:
            value=i.get_Q()
            if torch.is_tensor(value):
                value=value.item()
            values.append(value)
        print("children values: "+str(values)) 
        # print("*"*20)