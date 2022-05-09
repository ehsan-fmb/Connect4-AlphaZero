import math
from state import State
import numpy as np
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha=1


class AlphaZero_MCTS:
    def __init__(self, root, num_sim, brain,counter, exploration_weight=3, tau=0, cur_level=0):
        self.__root = root
        self.__exploration_weight = exploration_weight
        self.__sim_budget = num_sim
        self.__brain = brain
        self.__counter=counter
        self.__tau = tau
        self.__cur_level = cur_level

    def search(self):
        counter = 0
        start = time.time()
        while time.time() <= start + 3:
            cur = self.selection()
            if cur.get_N() < 1:
                delta = self.simulation(cur)
            else:
                self.expansion(cur)
                # no children condition should be checked!
                if len(cur.get_children()) > 0:
                    cur = self.best_child(cur)
                delta = self.simulation(cur)
            self.backpropagation(cur, delta)
            counter += 1

        # to check tau
        self.__cur_level = self.__cur_level + 1

        self.compute_mcts_policy()
        return self.__root.get_arow(), self.__root.get_acul()

    def compute_mcts_policy(self):
        children = self.__root.get_children()
        probs = []
        new_mcts = torch.zeros(7).to(device).detach()
        total = self.__root.get_N()- 1

        for i in range(len(children)):
            prob = children[i].get_N() / total
            new_mcts[children[i].get_acul()] = prob
            probs.append(prob)
        self.__root.change_mcts_policy(new_mcts)

        # compare the current level and the temperature
        if self.__cur_level < self.__tau:
            self.__root = np.random.choice(children, p=probs)
        else:
            self.__root = children[np.argmax(probs)]

    def backpropagation(self, cur, delta):
        terminal=self.__root.get_parent()
        while cur != terminal:
            cur.set_N(cur.get_N() + 1)
            cur.set_Q(cur.get_Q() + delta)
            delta = delta * -1
            cur = cur.get_parent()

    def expansion(self, cur):
        children = []
        if not cur.get_final():
            actions_priors = cur.get_inferences()[1]
            moves = cur.legal_moves()
            for i in moves:
                # get inferences from brain
                state = State(cur, False, None, -1 * cur.get_player(), actions_priors[0][i[1]], arow=i[0], acul=i[1])
                state.change_board(i[0], i[1])
                state.encode_board()
                action_probs, val = self.__brain(state.get_encoded_board())
                state.set_inferences(val, action_probs)
                children.append(state)
        cur.set_children(children)

    def selection(self):
        cur = self.__root
        while len(cur.get_children()) != 0:
            cur = self.best_child(cur)
        return cur

    def normalize(self, probs):
        total = sum(probs)
        if total != 1:
            probs[0] = probs[0] + 1 - total
        return probs

    def best_child(self, parent):
        c = self.__exploration_weight
        children = parent.get_children()

        # dirichlet noise
        priors=[]
        alphas=[]
        for i in range(len(children)):
            priors.append(float(children[i].get_prior_policy().cpu().detach().numpy()))
            alphas.append(alpha)
        noise=np.random.dirichlet(alphas)
        
        priors = [i * 0.8 for i in priors]
        priors=priors+0.2*noise
        priors=self.normalize(priors)

        pos = 0
        max_val = (children[pos].get_Q() / children[pos].get_N()) + c * priors[pos] *( math.sqrt(
            parent.get_N()) / children[pos].get_N())
        for i in range(1, len(children)):
            new_val = (children[i].get_Q() / children[i].get_N()) + c * priors[i] * (math.sqrt(
                parent.get_N()) / children[i].get_N())
            if new_val > max_val:
                pos = i
                max_val = new_val
        return children[pos]

    def simulation(self, cur):
        result=cur.evaluation()
        if result == "is not terminal":
            return cur.get_inferences()[0]
        else:
            cur.set_final()
            if result=="draw":
                return 0
            else:
                return 1

    def buffer(self, outcome):
        buffer=[]

        if outcome == "draw":
            z = torch.tensor([0]).to(device)
        else:
            z = torch.tensor([1]).to(device)

        cur=self.__root
        z = z.detach()

        while cur is not None:
            if cur.get_mcts_policy() is None:
                cur.change_mcts_policy(torch.zeros(7).to(device).detach())
    
            buffer.append([cur.get_encoded_board(),z,cur.get_mcts_policy()])
            
            # cur.plot()
            # print("outcome: "+str(z))
            # print("*"*20)

            cur = cur.get_parent()
            if cur is None:
                continue
            if cur.get_parent() is not None:
                z = z * -1
            else:
                z=torch.tensor([0]).to(device)
        
        # sys.exit()
        return buffer
    
    def change_root(self,row,cul):
        for child in self.__root.get_children():
            if child.get_arow()==row and child.get_acul()==cul:
                self.__root=child
                return

        if (len(self.__root.get_children())==0) and (self.__root.get_parent() is None):
            self.__root.set_player(-1 * self.__root.get_player())
            self.__root.change_board(row,cul)
