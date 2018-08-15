# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math
import copy

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);

            else:
                break;
        # returns random action from all the valid actions
        #print self.actionList
        print state.getGhostPositions()
        return self.actionList[0];



class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0, 5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        possible = state.getAllPossibleActions()
        for i in range(0, len(self.actionList)):
            self.actionList[i] = possible[random.randint(0, len(possible) - 1)]
        tempActionList = self.actionList[:]    #actionList is the action list with the highest score, tempActionList is to store the new action list obtained by mutating the best action list to check whether the new action list is better than best action list
        score= 0                                # to store the score for the tempActionList
        flag = 0
        highestScore= 0                         #to store the highest score of the actionList obtained so far
        while True:
            tempState= state
            for i in range(0,len(tempActionList)):
                ns = tempState.generatePacmanSuccessor(tempActionList[i])    #ns = new state
                if ns == None:                  #if else to check whether the new generated state is not a terminal state.
                    flag = 1
                    break
                elif ns.isWin() or ns.isLose():
                    tempState = ns
                    break
                else:
                    tempState = ns

            if flag != 1:                       #calculate the score of the action list and store the best action list.
                score = scoreEvaluation(tempState)
                if score >= highestScore:
                    highestScore = score
                    self.actionList = tempActionList[:]

            else:
                break

            for i in range(0, len(self.actionList)):        #mutate the best action list with some probability
                r = random.randint(0,10)
                if r >= 5:
                    tempActionList[i]= possible[random.randint(0, len(possible)-1)]
                else:
                    tempActionList[i]= self.actionList[i]

        return self.actionList[0]


class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        parents=[]
        nextGen=[]
        parentCount = 8
        lenActionList = 5
        tempState = state
        flag = 0
        rank = []
        parentScore = [0]*8
        possible = state.getAllPossibleActions()
        for i in range(0, parentCount):
            sl = []
            for j in range(0, 5):                           # 5= lenActionList
                sl.append(possible[random.randint(0, len(possible)-1)])
            parents.append(list(sl))

        while True:
            parentScore = [0]*8
            for i in range(0, 8):                # 8 = parentCount
                tempState= state
                for j in range(0, 5):                       #5 = lenActionList
                    ns = tempState.generatePacmanSuccessor(parents[i][j])
                    if ns == None:
                        flag = 1
                        break
                    elif ns.isWin() or ns.isLose():
                        tempState = ns
                        break
                    else:
                        tempState = ns
                if flag == 1:
                    break
                else:
                    parentScore[i] = scoreEvaluation(tempState)
            if flag == 1:
                break
            #CROSSOVER
            else:

                rank = self.ranking(parentScore)
                prob = self.probability(rank)
                for i in range(0, 4):
                    p1 = self.random_pick(parents, prob)
                    p2 = self.random_pick(parents, prob)
                    r = random.randint(0,10)
                    if r <= 7:
                        nextGen.append(list(p1))
                        nextGen.append(list(p2))
                    else:
                        l1 =[]
                        l2 =[]
                        for i in range(0, lenActionList):
                            r = random.randint(0, 10)
                            if r < 5:
                                l1.append(p1[i])
                                l2.append(p2[i])
                            else:
                                l1.append(p2[i])
                                l2.append(p1[i])

                        nextGen.append(l1)
                        nextGen.append(l2)
                #MUTATION
                for i in range(0, len(nextGen)):
                    r = random.randint(0,10)
                    if r <= 1:
                        rand = random.randint(0,lenActionList-1)
                        action = possible[random.randint(0, len(possible) - 1)]
                        nextGen[i][rand] = action
                parents = nextGen[:]

        parents = nextGen[:]
        highestScore = max(parentScore)
        i = parentScore.index(highestScore)

        return parents[i][0]



    def probability(self,rank):
        """
        function to calculate the probability of each action list
        :param rank: list of parent action list scores
        :return: the probability of each action list
        """
        prob = []
        r = rank
        for i in range(0, len(r)):
            val = r[i]/36.0
            prob.append(val)
        return prob

    def random_pick(self, parents, probabilities):
        """
        function to pick two action list using weighted ranking
        :param parents: the parent action lists
        :param probabilities: the probabilities associated with each action list
        :return: a action list
        """
        x = random.uniform(0, 1)
        l = parents
        pro = probabilities
        cumulative_probability = 0.0
        for item, item_probability in zip(l, pro):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item

    def ranking(self,parentScore):          #function to rank each action list
        #print "p= ", parentScore
        rank = []
        ps = parentScore
        sorting = sorted(ps)
        for ele in ps:
            i = sorting.index(ele)
            rank.append(i+1)
        return rank


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        root = Node()
        rootState = state
        action = []
        la = state.getLegalPacmanActions()
        for ele in la:
            newNode = Node(root, ele)
            root.addChild(newNode)
        while True:

            vNode, vState = self.treePolicy(root, rootState)
            if (vState != None):
                delta = self.defaultPolicy(rootState, vState)
                self.backUp(vNode, delta)
            else:
                break

        for i in range(0, len(root.child)):
            action.append((root.child[i].n, root.child[i].incomingAction))

        bestAction = max(action, key=lambda x: x[0])
        return bestAction[1]

    def treePolicy(self, v, state):
        """
        function to check whether the child of the node has been visited or not.
        if all the child has been visited then call bestChild() function, else
        expand the current node.
        :param v: current node
        :param state: root state
        :return: the state of the current node
        """
        ts = state
        flag = 0
        ele = state
        while ele!= None:
            # if len(v.child) != 0:
            for ele in v.child:
                if ele.n == 0:
                    e = self.Expand(ele, ts)
                    if e == None:
                        return (None, None)
                    else:
                        return (ele, e)  # (v, self.Expand(ele, ts))
                else:
                    flag = 1

            if flag == 1:
                g = self.bestChild(v)
                if g!= 0:
                    v = g
                else:
                    break
            # else:
            #     break

        return (None, None)

    def Expand(self, v, state):
        """
        function to add new child nodes to the current node and generate states from root to the current node.
        :param v: current node for which the child node has to be generated
        :param state: root state
        :return: state of the current node
        """
        sv = copy.deepcopy(v)
        tempState = state
        path = []
        while sv != None:
            if sv.incomingAction != None:
                path.append(sv.incomingAction)
            sv = sv.parent
        path.reverse()
        for a in path:
            ns = tempState.generatePacmanSuccessor(a)
            if ns == None:
                return None
            elif ns.isWin() or ns.isLose():
                tempState = ns
                break
            else:
                tempState = ns

        la = tempState.getLegalPacmanActions()
        for ele in la:
            newNode = Node(v, ele)
            v.addChild(newNode)

        return tempState

    def bestChild(self, v):
        """
        function to return the best child node using the UCT formula
        :param v: the node for which the best child is to be chosen
        :return: best child node, i.e., the child node with the maximum value of UCT
        """
        #best = -1000
        l = len(v.child)
        highUCT = -10000
        if l!= 0:
            for i in range(0, l):
                uct = ((v.child[i].q) / (v.child[i].n)) + math.sqrt((2 * math.log(v.n, 2)) / (v.child[i].n))
                #print uct
                if uct >= highUCT:
                    highUCT = uct
                    best = v.child[i]
            return best
        else:
            return 0

    def defaultPolicy(self, rs, v):
        """
        function to randomly rollout
        :param rs: root state
        :param v: current state
        :return: normalized score evaluation
        """
        vs = copy.deepcopy(v)
        act = vs.getLegalPacmanActions()
        if len(act) != 0:
            for i in range(0, 5):
                # act = vs.getLegalPacmanActions()
                ns = vs.generatePacmanSuccessor(act[random.randint(0, len(act) - 1)])
                if ns == None:
                    #print None
                    break
                elif ns.isLose() or ns.isWin():
                    vs = ns
                    #print "in def policy win lose"
                    break
                else:
                    vs = ns
                    act = vs.getLegalPacmanActions()
        score = normalizedScoreEvaluation(rs, vs)
        return score

    def backUp(self, vn, d):
        """
        function to back propopagate the value upto the root node
        :param vn: current node
        :param d:  normalized score evaluation
        :return:   none
        """
        while vn != None:
            vn.q = vn.q + d
            vn.n = vn.n + 1
            vn = vn.parent


class Node(object):
    def __init__(self, parent= None, action = None, q=0, n=0): #constructor
        self.parent = parent
        self.q = q
        self.n = n
        self.incomingAction = action
        self.child = []

    def addChild(self,obj):       #function to add child node to the parent node
        self.child.append(obj)



