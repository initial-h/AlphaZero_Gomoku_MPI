# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:19:11 2018

@author: initial-h
"""

import numpy as np
import copy
from operator import itemgetter
from collections import defaultdict


def rollout_policy_fn(board):
    '''
    a coarse, fast version of policy_fn used in the rollout phase.
    '''
    action_probs = np.random.rand(len(board.availables)) # rollout randomly
    return zip(board.availables, action_probs)

def policy_value_fn(board):
    '''
    a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state
    '''
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0

class TreeNode(object):
    '''
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    '''

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p # its the prior probability that action's taken to get this node

    def expand(self, action_priors):
        '''
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
        according to the policy function.
        '''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
        # expand all children that under this state

    def select(self, c_puct):
        '''
        Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        '''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
        # self._children is a dict
        # act_node[1].get_value will return the action with max Q+u and corresponding state

    def update(self, leaf_value):
        '''
        Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's perspective.
        '''
        self._n_visits += 1
        # update visit count
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # update Q, a running average of values for all visits.
        # there is just: (v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)

    def update_recursive(self, leaf_value):
        '''
        Like a call to update(), but applied recursively for all ancestors.
        '''
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
            # every step for revursive update,
            # we should change the perspective by the way of taking the negative
        self.update(leaf_value)

    def get_value(self, c_puct):
        '''
        Calculate and return the value for this node.
        It is a combination of leaf evaluations Q,
        and this node's prior adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
        value Q, and prior probability P, on this node's score.
        '''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        '''
        check if it's leaf node (i.e. no nodes below this have been expanded).
        '''
        return self._children == {}

    def is_root(self):
        '''
        check if it's root node
        '''
        return self._parent is None

class MCTS(object):
    '''
    A simple implementation of Monte Carlo Tree Search.
    '''
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        '''
        self._root = TreeNode(parent=None, prior_p=1.0)
        # root node do not have parent ,and sure with prior probability 1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout # times of tree search

    def _playout(self, state):
        '''
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        '''
        node = self._root
        while(1):
            # select action in tree
            if node.is_leaf():
                # break if the node is leaf node
                # print('breaking...................................')
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            # print('select action is ...',action)
            # print(action,state.availables)
            state.do_move(action)
            # this state should be the same state with current node

        action_probs, _ = self._policy(state)
        # Check for end of game
        end, winner = state.game_end()
        if not end:
            # expand the node
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
        # print('after update...', node._n_visits, node._Q)

    def _evaluate_rollout(self, state, limit=1000):
        '''
        Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        '''
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            # itemgetter
            # https://www.cnblogs.com/zhoufankui/p/6274172.html
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        # print('winner is ...',winner)
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        '''
        Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        '''
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            # use deepcopy and playout on the copy state

        # some statistics just for check
        # visits_count = defaultdict(int)
        # visits_count_dic = defaultdict(int)
        # self.sum = 0
        # Q_U_dic = defaultdict(int)
        # for act,node in self._root._children.items():
        #     visits_count[act] += node._n_visits
        #     visits_count_dic[str(state.move_to_location(act))] += node._n_visits
        #     self.sum += node._n_visits
        #     Q_U_dic[str(state.move_to_location(act))] = node.get_value(5)

        # print(Q_U_dic)
        # print(self.sum,visits_count_dic)

        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        '''
        Step forward in the tree, keeping everything we already know about the subtree.
        '''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    '''
    AI player based on MCTS
    '''
    def __init__(self, c_puct=5, n_playout=400):
        '''
        init a mcts class
        '''
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        '''
        set player index
        '''
        self.player = p

    def reset_player(self):
        '''
        reset player
        '''
        self.mcts.update_with_move(-1) # reset the node

    def get_action(self, board,is_selfplay=False,print_probs_value=0):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        '''
        sensible_moves = board.availables
        if board.last_move!=-1:
            self.mcts.update_with_move(last_move=board.last_move)
            # reuse the tree
            # retain the tree that can continue to use
            # so update the tree with opponent's move and do mcts from the current node

        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(move)
            # every time when get a move, update the tree
        else:
            print("WARNING: the board is full")

        return move, None

    def __str__(self):
        return "MCTS {}".format(self.player)








