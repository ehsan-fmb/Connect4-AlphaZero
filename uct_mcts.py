import math
import time
from state import State
import random
import numpy as np

# simulation1: randomly choose an action at each step
# simulation2: Use a simple heurisitc to evaluate the moves in rollouts(heavy playouts)

class UCT_MCTS:
    def __init__(self, root,simulation,t,exploration_weight=math.sqrt(3)):
        self.__root = root
        self.__exploration_weight = exploration_weight
        self.__simulation=simulation
        self.__time=t

    def search(self):
        start = time.time()
        counter=0
        while time.time() <= start + self.__time:
            cur = self.selection()
            if cur.get_N()<1:
                delta = self.__simulation(self,cur)
            else:
                self.expansion(cur)
                # no children condition should be checked!
                if len(cur.get_children())>0:
                    cur=cur.get_children()[0]
                delta=self.__simulation(self,cur)
            self.backpropagation(cur,delta)
            counter+=1

        # children = self.__root.get_children()
        # for i in range(len(children)):
        #     new_val=(children[i].get_Q() / children[i].get_N())
        #     print("value: "+str(new_val)+"  arow: "+str(children[i].get_arow())+"   acul: "+str(children[i].get_acul()))
        # print("********************************")
        best_child=self.best_child(self.__root,c=0)
        self.__root=best_child
        # print("player:" + str(self.__root.get_player()))
        # print("number of simulations: " + str(counter))
        # print("****************************")
        return best_child.get_arow(),best_child.get_acul()


    def backpropagation(self,cur,delta):
        while cur!=self.__root:
            cur.set_N(cur.get_N()+1)
            cur.set_Q(cur.get_Q()+delta)
            delta=delta*-1
            cur=cur.get_parent()


    def expansion(self,cur):
        children = []
        moves=cur.legal_moves()
        for i in moves:
            state=State(cur,False,None,cp=-1*cur.get_player(),prior_policy=0,arow=i[0],acul=i[1])
            state.change_board(i[0],i[1])
            children.append(state)
        random.shuffle(children)
        cur.set_children(children)

    def selection(self):
        cur = self.__root
        while len(cur.get_children()) != 0:
            cur = self.best_child(cur)
        return cur

    def best_child(self, parent,c=None):
        if c is None:
            c=self.__exploration_weight
        children = parent.get_children()
        pos = 0
        max_val = (children[pos].get_Q() / children[pos].get_N()) + c * math.sqrt(
            (2 * parent.get_N()) / children[pos].get_N())
        for i in range(1,len(children)):
            new_val=(children[i].get_Q() / children[i].get_N()) + c * math.sqrt(
            (2 * parent.get_N()) / children[i].get_N())
            if new_val> max_val:
                pos=i
                max_val=new_val
        return children[pos]

    def simulation1(self,cur):
        state=State(None,True,cur.get_board(),cur.get_player(),0)
        result=state.evaluation1()
        while result == "is not terminal":
            moves = state.legal_moves()
            random.shuffle(moves)
            state.set_player(-1 * state.get_player())
            state.change_board(moves[0][0],moves[0][1])
            result=state.evaluation1(cp=-1*cur.get_player())
        if result=="win":
            return 1
        elif result=="loss":
            return -1
        else:
            return 0

    def simulation2(self,cur):
        state = State(None, True, cur.get_board(), cur.get_player())
        result = state.evaluation1()
        while result == "is not terminal":
            state.set_player(-1 * state.get_player())
            moves = state.legal_moves()
            probabilities=state.heuristic(moves)
            choice=np.random.choice(np.arange(len(moves)),1,p=probabilities)
            state.change_board(moves[choice[0]][0], moves[choice[0]][1])
            result = state.evaluation1(cp=cur.get_player())

        if result == "win":
            return 1
        elif result == "loss":
            return -1
        else:
            return 0

    def change_root(self,row,cul):
        for child in self.__root.get_children():
            if child.get_arow()==row and child.get_acul()==cul:
                self.__root=child
                return

        if (len(self.__root.get_children())==0) and (self.__root.get_parent() is None):
            self.__root.set_player(-1 * self.__root.get_player())
            self.__root.change_board(row,cul)
