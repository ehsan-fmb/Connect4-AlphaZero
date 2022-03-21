import math
from state import State
import numpy as np
import torch

class MCTS:
    def __init__(self, root,num_sim,brain,exploration_weight=1.25):
        self.__root = root
        self.__exploration_weight = exploration_weight
        self.__sim_budget=num_sim
        self.__brain=brain

    def search(self):
        counter=0
        while counter <= self.__sim_budget:
            cur = self.selection()
            if cur.get_N()<1:
                delta = self.simulation(cur)
            else:
                self.expansion(cur)
                # no children condition should be checked!
                if len(cur.get_children())>0:
                    cur=self.best_child(cur)
                delta=self.simulation(cur)
            self.backpropagation(cur,delta)
            counter+=1

        self.compute_mcts_policy()
        return self.__root.get_arow(),self.__root.get_acul()


    def compute_mcts_policy(self):
        children=self.__root.get_children()
        probs=[]
        # the noise should be considered
        #noise=np.random.dirichlet(len(children))
        total=self.__root.get_N().item()-1
        for i in range(len(children)):
            prob=children[i].get_N().item()/total
            self.__root.change_mcts_policy(children[i].get_acul(),prob)
            probs.append(prob)
        self.__root= np.random.choice(children, p=probs)

    def backpropagation(self,cur,delta):
        while cur is not None:
            cur.set_N(cur.get_N()+1)
            cur.set_Q(cur.get_Q()+delta)
            delta=delta*-1
            cur=cur.get_parent()


    def expansion(self,cur):
        children = []
        actions_priors=cur.get_inferences()[1]
        moves=cur.legal_moves()
        for i in moves:
            #get inferences from brain
            state=State(cur,False,None,-1*cur.get_player(),actions_priors[i[1]],arow=i[0],acul=i[1])
            state.change_board(i[0],i[1])
            action_probs,val = self.__brain.predict(state.get_board())
            state.set_inferences(val,action_probs)
            children.append(state)
        cur.set_children(children)

    def selection(self):
        cur = self.__root
        while len(cur.get_children()) != 0:
            cur = self.best_child(cur)
        return cur

    def best_child(self, parent):
        c = self.__exploration_weight
        children = parent.get_children()
        pos = 0
        max_val = (children[pos].get_Q() / children[pos].get_N()) + c *children[pos].get_prior_policy()* math.sqrt(
            parent.get_N() / children[pos].get_N())
        for i in range(1,len(children)):
            new_val=(children[i].get_Q() / children[i].get_N()) + c*children[i].get_prior_policy() * math.sqrt(
            parent.get_N() / children[i].get_N())
            if new_val> max_val:
                pos=i
                max_val=new_val
        return children[pos]

    def simulation(self,cur):
        result=cur.evaluation()
        if result != "is not terminal":
            return cur.get_inferences()[0]
        if result=="win":
            return 1
        elif result=="loss":
            return -1
        else:
            return 0


    def buffer(self,outcome):
        value_loss=0
        policy_buffer=[]
        counter=0
        if outcome=="draw":
            z=torch.tensor(0)
        else:
            z=torch.tensor(1)

        cur=self.__root.get_parent()
        #bring all stuff in gpu
        z=z.to("cuda")
        while cur is not None:
            val,actions_probs=cur.get_inferences()
            policy_buffer.append(-torch.dot(torch.log(actions_probs).cuda(),cur.get_mcts_policy().cuda()))
            value_loss=torch.pow(z-val,2)+value_loss
            z=z*-1
            cur=cur.get_parent()
            counter+=1
        return value_loss,sum(policy_buffer),counter