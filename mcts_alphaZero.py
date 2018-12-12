# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:05:17 2018

@author: initial
"""


import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    # https://mp.weixin.qq.com/s/2xYgaeLlmmUfxiHCbCa8dQ
    # avoid float overflow and underflow
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    '''
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    '''

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p # its the prior probability that action's taken to get this node

    def expand(self, action_priors,add_noise):
        '''
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        '''
        # when train by self-play, add dirichlet noises in each node

        # should note it's different from paper that only add noises in root node
        # i guess alphago zero discard the whole tree after each move and rebuild a new tree, so it's no conflict
        # while here i contained the Node under the chosen action, it's a little different.
        # there's no idea which is better
        # in addition, the parameters should be tried
        # for 11x11 board,
        # dirichlet parameter :0.3 is ok, should be smaller with a bigger board,such as 20x20 with 0.03
        # weights between priors and noise: 0.75 and 0.25 in paper and i don't change it here,
        # but i think maybe 0.8/0.2 or even 0.9/0.1 is better because i add noise in every node
        # rich people can try some other parameters
        if add_noise:
            action_priors = list(action_priors)
            length = len(action_priors)
            dirichlet_noise = np.random.dirichlet(0.3 * np.ones(length))
            for i in range(length):
                if action_priors[i][0] not in self._children:
                    self._children[action_priors[i][0]] = TreeNode(self,0.75*action_priors[i][1]+0.25*dirichlet_noise[i])
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        '''
        Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        '''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        '''
        Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        '''
        self._n_visits += 1
        # update visit count
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # Update Q, a running average of values for all visits.
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
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        '''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        '''
        check if leaf node (i.e. no nodes below this have been expanded).
        '''
        return self._children == {}

    def is_root(self):
        '''
        check if it's root node
        '''
        return self._parent is None


class MCTS(object):
    '''
    An implementation of Monte Carlo Tree Search.
    '''
    def __init__(self, policy_value_fn,action_fc,evaluation_fc, is_selfplay,c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        '''
        self._root = TreeNode(None, 1.0)
        # root node do not have parent ,and sure with prior probability 1

        self._policy_value_fn = policy_value_fn
        self._action_fc = action_fc
        self._evaluation_fc = evaluation_fc

        self._c_puct = c_puct
        # it's 5 in paper and don't change here,but maybe a better number exists in gomoku domain
        self._n_playout = n_playout # times of tree search
        self._is_selfplay = is_selfplay

    def _playout(self, state):
        '''
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        '''
        node = self._root
        # print('============node visits:',node._n_visits)
        # deep = 0
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            # print('move in tree...',action)
            state.do_move(action)
            # deep+=1
        # print('-------------deep is :',deep)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy_value_fn(state,self._action_fc,self._evaluation_fc)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            # print('expand move:',state.width*state.height-len(state.availables),node._n_visits)
            node.expand(action_probs,add_noise=self._is_selfplay)
        else:
            # for end state，return the "true" leaf_value
            # print('end!!!',node._n_visits)
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
        # no rollout here

    def get_move_visits(self, state):
        '''
        Run all playouts sequentially and return the available actions and
        their corresponding visiting times.
        state: the current game state
        '''
        for n in range(self._n_playout):
            # print('playout:',n)
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        return acts, visits

    def update_with_move(self, last_move):
        '''
        Step forward in the tree, keeping everything we already know
        about the subtree.
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
    def __init__(self, policy_value_function,action_fc,evaluation_fc,c_puct=5, n_playout=400, is_selfplay=0):
        '''
        init some parameters
        '''
        self._is_selfplay = is_selfplay
        self.policy_value_function = policy_value_function
        self.action_fc = action_fc
        self.evaluation_fc = evaluation_fc
        self.first_n_moves = 12
        # For the first n moves of each game, the temperature is set to τ = 1,
        # For the remainder of the game, an infinitesimal temperature is used, τ→ 0.
        # in paper n=30, here i choose 12 for 11x11, entirely by feel
        self.mcts = MCTS(policy_value_fn = policy_value_function,
                         action_fc = action_fc,
                         evaluation_fc = evaluation_fc,
                         is_selfplay = self._is_selfplay,
                         c_puct = c_puct,
                         n_playout = n_playout)

    def set_player_ind(self, p):
        '''
        set player index
        '''
        self.player = p

    def reset_player(self):
        '''
        reset player
        '''
        self.mcts.update_with_move(-1)

    def get_action(self,board,is_selfplay,print_probs_value):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        '''
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            if is_selfplay:
                acts, visits = self.mcts.get_move_visits(board)
                if board.width * board.height - len(board.availables) <= self.first_n_moves:
                    # For the first n moves of each game, the temperature is set to τ = 1
                    temp = 1
                    probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                    move = np.random.choice(acts, p=probs)
                else:
                    # For the remainder of the game, an infinitesimal temperature is used, τ→ 0
                    temp = 1e-3
                    probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                    move = np.random.choice(acts, p=probs)

                self.mcts.update_with_move(move)
                # update the tree with self move
            else:
                self.mcts.update_with_move(board.last_move)
                # update the tree with opponent's move and then do mcts from the new node

                acts, visits = self.mcts.get_move_visits(board)
                temp = 1e-3
                # always choose the most visited move
                probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                move = np.random.choice(acts, p=probs)

                self.mcts.update_with_move(move)
                # update the tree with self move

            p = softmax(1.0 / 1.0 * np.log(np.array(visits) + 1e-10))
            move_probs[list(acts)] = p
            # return the prob with temp=1

            if print_probs_value and move_probs is not None:
                act_probs, value = self.policy_value_function(board,self.action_fc,self.evaluation_fc)
                print('-' * 10)
                print('value',value)
                # print the probability of each move
                probs = np.array(move_probs).reshape((board.width, board.height)).round(3)[::-1, :]
                for p in probs:
                    for x in p:
                        print("{0:6}".format(x), end='')
                    print('\r')

            return move,move_probs

        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Alpha {}".format(self.player)


