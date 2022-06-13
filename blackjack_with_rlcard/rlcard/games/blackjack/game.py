from copy import deepcopy
import numpy as np

from rlcard.games.blackjack import Dealer
from rlcard.games.blackjack import Player
from rlcard.games.blackjack import Judger

class BlackjackGame:

    def __init__(self, allow_step_back=False):
        ''' Initialize the class Blackjack Game
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState() # ❓

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']

    def init_game(self):
        ''' Initialilze the game

        Returns:
            state (dict): the first state of the game
            player_id (int): current player's id
        '''
        self.dealer = Dealer(self.np_random) # 庄家初始化

        # 玩家初始化
        self.players = []
        for i in range(self.num_players):
            self.players.append(Player(i, self.np_random))

        # 裁判初始化
        self.judger = Judger(self.np_random)

        # 给玩家和庄家各发两张牌
        for i in range(2): # 先每人一张牌，再发第二轮
            for j in range(self.num_players): # 先发各玩家的牌，再发庄家
                self.dealer.deal_card(self.players[j])
            self.dealer.deal_card(self.dealer)

        # 检查各玩家初始发牌是否爆牌
        for i in range(self.num_players):
            self.players[i].status, self.players[i].score = self.judger.judge_round(self.players[i])
        # 检查庄家初始发牌是否爆牌
        self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)

        self.winner = {'dealer': 0}
        for i in range(self.num_players):
            self.winner['player' + str(i)] = 0

        self.history = []
        self.game_pointer = 0

        return self.get_state(self.game_pointer), self.game_pointer

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): a specific action of blackjack. (Hit or Stand)

        Returns:/
            dict: next player's state
            int: next plater's id
        '''
        if self.allow_step_back:
            p = deepcopy(self.players[self.game_pointer])
            d = deepcopy(self.dealer)
            w = deepcopy(self.winner)
            self.history.append((d, p, w))

        next_state = {}
        # Play hit
        if action != "stand":
            self.dealer.deal_card(self.players[self.game_pointer]) # 对应玩家获得发牌
            self.players[self.game_pointer].status, self.players[self.game_pointer].score = self.judger.judge_round(
                self.players[self.game_pointer]) # 获取对应玩家是否爆牌与手牌和
            if self.players[self.game_pointer].status == 'bust': # 玩家要牌后爆牌
                # game over, set up the winner, print out dealer's hand # If bust, pass the game pointer
                if self.game_pointer >= self.num_players - 1:
                    while self.judger.judge_score(self.dealer.hand) < 17: # 庄家手牌和小于 17 则继续要牌
                        self.dealer.deal_card(self.dealer)
                    self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer) # 获取庄家是否爆牌与手牌和
                    for i in range(self.num_players): # 给对应玩家游戏结果
                        self.judger.judge_game(self, i) 
                    self.game_pointer = 0
                else: # 切换另一玩家
                    self.game_pointer += 1

                
        elif action == "stand": # If stand, first try to pass the pointer, if it's the last player, dealer deal for himself, then judge game for everyone using a loop
            self.players[self.game_pointer].status, self.players[self.game_pointer].score = self.judger.judge_round(
                self.players[self.game_pointer])
            if self.game_pointer >= self.num_players - 1:
                while self.judger.judge_score(self.dealer.hand) < 17:
                    self.dealer.deal_card(self.dealer)
                self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)
                for i in range(self.num_players):
                    self.judger.judge_game(self, i) 
                self.game_pointer = 0
            else:
                self.game_pointer += 1


            
            

        hand = [card.get_index() for card in self.players[self.game_pointer].hand] # 获取对应玩家当前手牌

        if self.is_over(): # 判断游戏是否结束，结束则获取庄家全部手牌
            dealer_hand = [card.get_index() for card in self.dealer.hand]
        else: # 否则获取庄家第二张以后的手牌
            dealer_hand = [card.get_index() for card in self.dealer.hand[1:]]

        for i in range(self.num_players): # 存储各玩家手牌到 player0 hand ...
            next_state['player' + str(i) + ' hand'] = [card.get_index() for card in self.players[i].hand]
        next_state['dealer hand'] = dealer_hand # 存储庄家手牌到 dealer hand
        next_state['actions'] = ('hit', 'stand') # 存储两个合法动作
        next_state['state'] = (hand, dealer_hand) # 存储玩家和庄家手牌到 state

        

        return next_state, self.game_pointer

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            Status (bool): check if the step back is success or not
        '''
        #while len(self.history) > 0:
        if len(self.history) > 0:
            self.dealer, self.players[self.game_pointer], self.winner = self.history.pop()
            return True
        return False

    def get_num_players(self):
        ''' Return the number of players in blackjack

        Returns:
            number_of_player (int): blackjack only have 1 player
        '''
        return self.num_players

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            number_of_actions (int): there are only two actions (hit and stand)
        '''
        return 2

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            player_id (int): current player's id
        '''
        return self.game_pointer

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            state (dict): corresponding player's state
        '''
        '''
                before change state only have two keys (action, state)
                but now have more than 4 keys (action, state, player0 hand, player1 hand, ... , dealer hand)
                Although key 'state' have duplicated information with key 'player hand' and 'dealer hand', I couldn't remove it because of other codes
                To remove it, we need to change dqn agent too in my opinion
                '''
        state = {}
        state['actions'] = ('hit', 'stand') # 存储两个合法动作
        hand = [card.get_index() for card in self.players[player_id].hand] # 各玩家手牌
        if self.is_over(): # 庄家手牌（游戏未结束时只能看见庄家一张手牌）
            dealer_hand = [card.get_index() for card in self.dealer.hand]
        else:
            dealer_hand = [card.get_index() for card in self.dealer.hand[1:]]

        for i in range(self.num_players): # 存储各玩家手牌到 player0 hand ...
            state['player' + str(i) + ' hand'] = [card.get_index() for card in self.players[i].hand]
        state['dealer hand'] = dealer_hand # 存储庄家手牌到 dealer hand
        state['state'] = (hand, dealer_hand) # 存储玩家和庄家手牌到 state

        return state

    def is_over(self):
        ''' Check if the game is over

        Returns:
            status (bool): True/False
        '''
        '''
                I should change here because judger and self.winner is changed too
                '''
        for i in range(self.num_players):
            if self.winner['player' + str(i)] == 0:
                return False

        return True
