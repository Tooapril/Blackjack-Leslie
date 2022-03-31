from turtle import TPen, color
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout

stacks = 50000 # 牌局数目
players = 1 # 玩家数目
num_decks = 1 # 牌副数目

card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

# This function lists out all permutations（排列） of ace values in the array sum_array
# For example, if you have 2 aces, there are 4 permutations:
#   [[1,1], [1,11], [11,1], [11,11]]
# These permutations lead to 3 unique sums: [2, 12, 22]
# of these 3, only 2 are <=21 so they are returned: [2, 12]
def get_ace_values(temp_list):
    sum_array = np.zeros((2**len(temp_list), len(temp_list)))
    # This loop gets the permutations
    for i in range(len(temp_list)):
        n = len(temp_list) - i
        half_len = int(2**n * 0.5)
        for rep in range(int(sum_array.shape[0]/half_len/2)): #⭐️ shape[0] 返回 numpy 数组的行数
            sum_array[rep*2**n : rep*2**n+half_len, i] = 1
            sum_array[rep*2**n+half_len : rep*2**n+half_len*2, i] = 11
    # Only return values that are valid (<=21)
    # return list(set([int(s) for s in np.sum(sum_array, axis=1) if s<=21])) #⭐️ 将所有 'A' 能组成总和不超过 21 的值返回
    return [int(s) for s in np.sum(sum_array, axis=1)] #⭐️ 将所有 'A' 能组成的点数以 int 类型返回（有重复和超过 21 点的值）


# Convert num_aces, an int to a list of lists
# For example if num_aces=2, the output should be [[1,11],[1,11]]
# I require this format for the get_ace_values function
def ace_values(num_aces):
    temp_list = []
    for i in range(num_aces):
        temp_list.append([1,11])
    return get_ace_values(temp_list)


# Make a deck -- 根据给定副数洗好牌
def make_decks(num_decks, card_types):
    new_deck = []
    for i in range(num_decks):
        for j in range(4): # 代表黑红梅方
            new_deck.extend(card_types) #⭐️ extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
    random.shuffle(new_deck)
    return new_deck


# Total up value of hand
def total_up(hand):
    aces = 0 # 记录 ‘A’ 的数目
    total = 0 # 记录除 ‘A’ 以外数字之和
    
    for card in hand:
        if card != 'A':
            total += card
        else:
            aces += 1
            
    # Call function ace_values to produce list of possible values for aces in hand
    ace_value_list = ace_values(aces)
    final_totals = [i+total for i in ace_value_list if i+total<=21] # ‘A’ 可以是 1 也可以是 11，当前牌值不超过 21 时，取最大值 -- 规则❗️

    if final_totals == []:
        return min(ace_value_list) + total
    else:
        return max(final_totals)

 
# Given the relevant inputs, the function below uses the neural net to make a prediction
# and then based on that prediction, decides whether to hit or stay
# 将玩家当前手牌总数和、起手是否有‘A’、庄家起手是否有‘A’传入神经网络模型，如果预测结果大于 0.52，则 hit，否则 stand
def model_decision(model, player_sum, has_ace, dealer_card_num):
    # 将需要进入神经网络模型的数据统一格式
    # [[18  0  0  6]]
    input_array = np.array([player_sum, 0, has_ace, dealer_card_num]).reshape(1, -1) # 二维数组变成一行 (1, n)
    # print(input_array)
    
    # input_array 作为输入传入神经网络，使用预测函数后存入 predict_correct
    # [[0.10379896]]
    predict_correct = model.predict(input_array)
    # print(predict_correct)
    if predict_correct >= 0.52:
        return 1
    else:
        return 0
 
  
# Play a game of blackjack (after the cards are dealt)
def play_game(type, live_total, dealer_hand, player_hands, blackjack, current_player_results, dealer_cards, hit_stay=0, model=None):
    action = 0 # 标记玩家是否进行要牌操作
    
    # Dealer checks for 21
    if set(dealer_hand) == blackjack: # 庄家直接二十一点
        for player in range(players):
            if set(player_hands[player]) != blackjack: # 玩家此时不是二十一点，则结果为 -1 -- 规则❗️
                current_player_results[0, player] = -1
            else:
                current_player_results[0, player] = 0
    else: # 庄家不是二十一点，各玩家进行要牌、弃牌动作
        for player in range(players):
            # Players check for 21
            if set(player_hands[player]) == blackjack: # 玩家此时直接二十一点，则结果为 1
                current_player_results[0, player] = 1
            else: # 玩家也不是二十一点
                if type == 0: # Hit only when we know we will not bust -- 在玩家当前手牌点数不超过 11 时，才决定拿牌
                    while total_up(player_hands[player]) <= 11:
                        player_hands[player].append(dealer_cards.pop(0))
                        if total_up(player_hands[player]) > 21: # 拿完牌后再次确定是否爆牌，爆牌则结果为 -1
                            current_player_results[0, player] = -1
                            break
                elif type == 1: # Hit randomly, check for busts -- 在玩家当前手牌不是二十一点时，以 hit_stay 是否大于 0.5 的方式决定拿牌
                    if (hit_stay >= 0.5) and (total_up(player_hands[player]) != 21):
                        player_hands[player].append(dealer_cards.pop(0))
                        action = 1 #❓
                        live_total.append(total_up(player_hands[player])) # 玩家要牌后，将点数和记录到 live_total
                        if total_up(player_hands[player]) > 21: # 拿完牌后再次确定是否爆牌，爆牌则结果为 -1
                            current_player_results[0, player] = -1
                elif type == 2: # Neural net decides whether to hit or stay 
                    # -- 在玩家当前手牌不是二十一点时，通过玩家当前手牌、玩家和庄家起手是否有‘A’给神经网络计算后，大于 0.52，则继续拿牌
                    if 'A' in player_hands[player][0:2]: # 玩家起手有 ‘A’
                        ace_in_hand = 1
                    else:
                        ace_in_hand = 0
                    if dealer_hand[0] == 'A': # 庄家起手有 ‘A’
                        dealer_face_up_card = 11
                    else:
                        dealer_face_up_card = dealer_hand[0]
                    
                    while (model_decision(model, total_up(player_hands[player]), ace_in_hand, 
                                          dealer_face_up_card) == 1) and (total_up(player_hands[player]) != 21):
                        player_hands[player].append(dealer_cards.pop(0))
                        action = 1
                        live_total.append(total_up(player_hands[player]))
                        if total_up(player_hands[player]) > 21:
                            current_player_results[0, player] = -1
                            break
                
    
    # Dealer hits based on the rules
    while total_up(dealer_hand) < 17: # 庄家牌值小于 17，则继续要牌
        dealer_hand.append(dealer_cards.pop(0))
        
    # Compare dealer hand to players hand but first check if dealer busted
    if total_up(dealer_hand) > 21: # 庄家爆牌
        for player in range(players): # 将结果不是 -1 的各玩家设置结果为 1
            if current_player_results[0, player] != -1:
                current_player_results[0, player] = 1
    else: # 庄家没爆牌
        for player in range(players): # 将玩家牌点数大于庄家牌点数的玩家结果置为 1
            if total_up(player_hands[player]) > total_up(dealer_hand):
                if total_up(player_hands[player]) <= 21:
                    current_player_results[0, player] = 1
            elif total_up(player_hands[player]) == total_up(dealer_hand):
                current_player_results[0, player] = 0
            else:
                current_player_results[0, player] = -1
    
    return current_player_results, dealer_cards, live_total, action


def step(type, model=None):    
    dealer_card_feature = []
    player_card_feature = []
    player_live_total = []
    player_live_action = []
    player_results = []

    for stack in range(stacks):
        blackjack = set(['A', 10])
        dealer_cards = make_decks(num_decks, card_types) # 根据给定牌副数洗牌
        while len(dealer_cards) > 20: # 牌盒里的牌不大于 20 张就没必要继续用这副牌进行游戏 -- 规则⭐️
            
            current_player_results = np.zeros((1, players))
            
            dealer_hand = []
            player_hands = [[] for player in range(players)]
            live_total = []
            
            # Deal FIRST card
            for player, hand in enumerate(player_hands): # 先给所有玩家发第一张牌
                player_hands[player].append(dealer_cards.pop(0)) # 将洗好的牌分别发给玩家
            dealer_hand.append(dealer_cards.pop(0)) # 再给庄家发第一张牌
            # Deal SECOND card
            for player, hand in enumerate(player_hands): # 先给所有玩家发第二张牌
                player_hands[player].append(dealer_cards.pop(0)) # 接着刚刚洗好的牌继续发牌
            dealer_hand.append(dealer_cards.pop(0)) # 再给庄家发第二张牌
            
            if type == 0:
                current_player_results, dealer_cards, live_total, action = play_game(0, live_total, dealer_hand, player_hands, blackjack, 
                                                                                     current_player_results, dealer_cards)
            elif type == 1:
                # Record the player's live total after cards are dealt
                live_total.append(total_up(player_hands[player]))
                
                # 前 25000 局，玩家在发牌后手牌不是 21 点就继续拿牌；
                # 后 25000 局，玩家在发牌后手牌不是 21 点不继续拿牌。
                if stack < 25000:
                    hit_stay = 1
                else:
                    hit_stay = 0
                
                current_player_results, dealer_cards, live_total, action = play_game(1, live_total, dealer_hand, player_hands, blackjack, 
                                                                                     current_player_results, dealer_cards, hit_stay)
            elif type == 2:
                # Record the player's live total after cards are dealt
                live_total.append(total_up(player_hands[player]))
                
                current_player_results, dealer_cards, live_total, action = play_game(2, live_total, dealer_hand, player_hands, blackjack, 
                                                                                     current_player_results, dealer_cards, model=model)
            
            # Track features
            dealer_card_feature.append(dealer_hand[0]) # 将庄家的第一张牌存入新的 list
            player_card_feature.append(player_hands) # 将每个玩家当前手牌存入新的 list
            player_results.append(list(current_player_results[0])) # 将各玩家的输赢结果存入新的 list
            
            if type == 1 or type == 2:
                player_live_total.append(live_total) # 将 所有玩家发牌后的点数和 以及 采取要牌行动玩家的点数和 存入新的 list
                player_live_action.append(action) # 将玩家是否采取要牌行动存入新的 list（只要有一个玩家要牌，action = 1）
                    
    model_df = pd.DataFrame() # 构造数据集
    model_df['dealer_card'] = dealer_card_feature # 所有游戏庄家的第一张牌
    # model_df['player_total'] = [total_up(i[0]) for i in player_card_feature] # 所有游戏第一个玩家的点数和（第一个玩家 -- 作为数据分析对象❗️）
    model_df['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature] # 所有游戏第一个玩家前两张牌的点数和（第一个玩家 -- 作为数据分析对象❗️）
    model_df['Y'] = [i[0] for i in player_results] # 所有游戏第一个玩家输赢结果（第一个玩家 -- 作为数据分析对象❗️）
    
    if type == 1 or type == 2:
        model_df['hit?'] = player_live_action # 玩家在发牌后是否要牌
    
    has_ace = []
    for i in player_card_feature:
        if ('A' in i[0][0:2]): # 玩家一发牌有 ‘A’，has_ace 列表追加一个 1
            has_ace.append(1)
        else: # 玩家一发牌无 ‘A’，has_ace 列表追加一个 0
            has_ace.append(0)
    model_df['has_ace'] = has_ace
    
    dealer_card_num = []
    for i in model_df['dealer_card']:
        if i == 'A': # 庄家第一张牌是 ‘A’，dealer_card_num 列表追加一个 11
            dealer_card_num.append(11)
        else: # 庄家第一张牌不是 ‘A’，dealer_card_num 列表追加该值
            dealer_card_num.append(i)
    model_df['dealer_card_num'] = dealer_card_num  
    
    lose = []
    for i in model_df['Y']:
        if i == -1: # 玩家输，lose 列表追加一个 1，e.g. [1, 1, ...]
            lose.append(1)
        else: # 玩家平局或赢，lose 列表追加一个 0，e.g. [0, 0, ...]
            lose.append(0)
    model_df['lose'] = lose

    if type == 1:
        # 如果玩家要牌且输了，那么不要是正确的决定；
        # 如果玩家不动且输了，那么要牌是正确的决定；
        # 如果玩家要牌且未输，那么要牌是正确的决定；
        # 如果玩家不动且未输，那么不要是正确的决定。
        correct = []
        for i, val in enumerate(model_df['lose']):
            if val == 1: # 玩家输
                if player_live_action[i] == 1: # 玩家采取要牌动作（玩家一输了 val = 1，玩家二采取了要牌动作 action = 1 有什么关系❓）
                    correct.append(0)
                else:
                    correct.append(1)
            else:
                if player_live_action[i] == 1:
                    correct.append(1)
                else:
                    correct.append(0)
        model_df['correct_action'] = correct
        
    
    # 统计玩家一的所有输、赢、平的次数
    # -1.0    199610
    #  1.0     99685
    #  0.0     13289
    # Name: 0, dtype: int64 
    # 312584
    count = pd.DataFrame(player_results)[0].value_counts()
    print(count, sum(count))

    return model_df


# Train a neural net to play blackjack
def train(model_df):
    # Set up variables for neural net
    feature_list = [i for i in model_df.columns if i not in ['dealer_card', 'Y', 'lose', 'correct_action']]
    # print(feature_list)
    
    # 将模型里的数据按矩阵形式存储
    train_X = np.array(model_df[feature_list]) 
    train_Y = np.array(model_df['correct_action']).reshape(-1, 1) # 二维数组变成一列 (n, 1)
    # print(train_X, train_Y)
    
    # Set up a neural net with 5 layers
    model = Sequential()
    model.add(Dense(16))
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(8))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit(train_X, train_Y, epochs=20, batch_size=256, verbose=1)

    # train_X 作为输入传入神经网络，使用预测函数后存入 pre_Y_train
    # train_Y 作为输出实际值，转变格式后存入 actuals
    # [[0.4260913 ]
    #  [0.3595919 ]
    #  [0.24476886]
    #  ...
    #  [0.2946579 ]
    #  [0.39343864]
    #  [0.27353495]] 
    # [1 0 0 ... 0 1 0]
    pred_Y_train = model.predict(train_X)
    actuals = train_Y[:, -1] # 将二维数组将为一维
    # print(pred_Y_train, actuals)
    
    return model, pred_Y_train, actuals

# 绘制多模型数据分析图
def comparison_chart(data, position):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x=data.index-0.28, height=data['naive'].values, color='blue', width=0.3, label='Naive')
    ax.bar(x=data.index, height=data['random'].values, color='orange', width=0.3, label='Random')
    ax.bar(x=data.index+0.28, height=data['smart'].values, color='red', width=0.3, label='Smart')
    ax.set_xlabel(position.capitalize() + "'s Card", fontsize=16) # capitalize() -- 将字符串首字母大写
    ax.set_ylabel('Probability of Tie or Win', fontsize=16)
    plt.xticks(np.arange(2, 12, 1.0))

    plt.legend()
    plt.tight_layout()
    plt.savefig(fname= './img/' + position + '_card_probs_smart', dpi=150)

# 多个模型数据分析
def comparison(model_df_naive, model_df_random, model_df_smart):
    data = pd.DataFrame()
    
    # 模型对比：查看庄家第一张牌牌值对玩家“不输”的影响
    # 保守模型
    data_naive = 1 - (model_df_naive.groupby(by='dealer_card_num').sum()['lose'] / 
                        model_df_naive.groupby(by='dealer_card_num').count()['lose'])
    # 随机模型
    data_random = 1 - (model_df_random.groupby(by='dealer_card_num').sum()['lose'] / 
                        model_df_random.groupby(by='dealer_card_num').count()['lose'])
    # 新模型
    data_smart = 1 - (model_df_smart.groupby(by='dealer_card_num').sum()['lose'] / 
                        model_df_smart.groupby(by='dealer_card_num').count()['lose'])
    
    data['naive'] = data_naive
    data['random'] = data_random
    data['smart'] = data_smart
    comparison_chart(data, 'dealer')
    
    # 模型对比：查看玩家前两张初始牌值对玩家“不输”的影响
    # 保守模型
    data_naive = 1 - (model_df_naive.groupby(by='player_total_initial').sum()['lose'] / 
                        model_df_naive.groupby(by='player_total_initial').count()['lose'])
    # 随机模型
    data_random = 1 - (model_df_random.groupby(by='player_total_initial').sum()['lose'] / 
                        model_df_random.groupby(by='player_total_initial').count()['lose'])
    # 新模型
    data_smart = 1 - (model_df_smart.groupby(by='player_total_initial').sum()['lose'] / 
                        model_df_smart.groupby(by='player_total_initial').count()['lose'])

    data['naive'] = data_naive[:-1]
    data['random'] = data_random[:-1]
    data['smart'] = data_smart[:-1]
    comparison_chart(data, 'player')
    
    # round(x, [, n]) 返回浮点数 x 四舍五入值，小数点后保留 n 位
    # Random: 0.3048
    # Random: 0.4094
    # Random: 0.3838
    print('Random: ' + str(round(model_df_random[model_df_random['Y']==1].shape[0]/model_df_random.shape[0], 4)))
    print('Random: ' + str(round(model_df_naive[model_df_naive['Y']==1].shape[0]/model_df_naive.shape[0], 4)))
    print('Random: ' + str(round(model_df_smart[model_df_smart['Y']==1].shape[0]/model_df_smart.shape[0], 4)))

# 单个模型数据分析
def presentation(type, model_df):
    if type == 1:
        # 查看庄家不同手牌对玩家“不输”的影响
        data = 1 - (model_df.groupby(by='dealer_card').sum()['lose'] / 
                    model_df.groupby(by='dealer_card').count()['lose'])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.barplot(x=data.index, y=data.values)
        ax.set_xlabel("Dealer's Card", fontsize=16)
        ax.set_ylabel("Probability of Tie or Win", fontsize=16)

        plt.tight_layout()
        plt.savefig(fname='./img/dealer_card_probs', dpi=150)


        # 查看玩家前两张发牌点数和对玩家“不输”的影响
        data = 1 - (model_df.groupby(by='player_total_initial').sum()['lose'] / 
                    model_df.groupby(by='player_total_initial').count()['lose'])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.barplot(x=data[:-1].index, y=data[:-1].values)
        ax.set_xlabel("Player's Hand Value", fontsize=16)
        ax.set_ylabel("Probability of Tie or Win", fontsize=16)

        plt.tight_layout()
        plt.savefig(fname='./img/player_hand_probs', dpi=150)


        # 查看玩家有‘A’对玩家“输”的影响
        # has_ace
        # 0    0.683229
        # 1    0.384232
        # Name: lose, dtype: float64
        # print(model_df.groupby(by='has_ace').sum()['lose'] / 
        #     model_df.groupby(by='has_ace').count()['lose'])

        # 去掉玩家初始 21 点的情况后，初始手牌与庄家第一张牌对玩家“不输”的影响
        pivot_data = model_df[model_df['player_total_initial'] != 21] # 去掉玩家一初始手牌点数为 21 的数据
        losses_pivot = pd.pivot_table(pivot_data, values='lose', index=['dealer_card_num'], 
                                    columns=['player_total_initial'], aggfunc=np.sum)
        games_pivot = pd.pivot_table(pivot_data, values='lose', index=['dealer_card_num'], 
                                    columns=['player_total_initial'], aggfunc='count')
        heat_data = 1 - losses_pivot.sort_index(ascending=False) / games_pivot.sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(heat_data, square=False, cmap="PiYG")
        ax.set_xlabel("Player's Hand Value", fontsize=16)
        ax.set_ylabel("Dealer's Card", fontsize=16)

        plt.savefig(fname='./img/heat_map_random', dpi=150)


        # # 去掉玩家初始 21 点的情况后，玩家手牌与庄家第一张牌对玩家“不输”的影响
        # model_tmp = model_df[model_df['player_total_initial'] != 21] # 去掉玩家一初始手牌点数为 21 的数据
        # pivot_data2 = model_tmp[model_tmp['player_total'] < 21]
        # losses_pivot2 = pd.pivot_table(pivot_data2, values='lose', index=['dealer_card_num'], 
        #                             columns=['player_total'], aggfunc=np.sum)
        # games_pivot2 = pd.pivot_table(pivot_data2, values='lose', index=['dealer_card_num'], 
        #                             columns=['player_total'], aggfunc='count')
        # heat_data2 = 1 - losses_pivot2.sort_index(ascending=False) / games_pivot2.sort_index(ascending=False)

        # fig, ax = plt.subplots(figsize=(16, 8))
        # sns.heatmap(heat_data2, square=False, cmap="PiYG")
        # ax.set_xlabel("Player's Hand Value", fontsize=16)
        # ax.set_ylabel("Dealer's Card", fontsize=16)

        # plt.savefig(fname='./img/heat_map_random2', dpi=150)
        
    elif type == 2:
        # Check out the probability of hitting for various player hand totals
        # -- 查看玩家前两张点数和对玩家要牌‘输’的影响
        # 以玩家前两张牌点数和分组，统计玩家输的局数，将数据转为 np.array()
        # [ 20  35  56  97  96 109 153 157 346 344 320 276 264 270 229 220 327 144]
        game_count_df = model_df.groupby(by=['player_total_initial']).count()['lose']
        game_count = np.array(game_count_df)
        # print(game_count_df, game_count)
        
        # 在要牌的所有数据中，以玩家前两张牌点数和分组，统计玩家输的局数，将数据转为 list()
        hit_count = list(model_df[model_df['hit?']==1].groupby(by=['player_total_initial']).count()['lose'])
        # print(model_df[model_df['hit?']==1].groupby(by=['player_total_initial']).count()['lose'], hit_count)
        hit_count.extend([0 for i in range(len(game_count) - len(hit_count))])
        hit_rate_df = pd.DataFrame(np.array(hit_count) / np.array(game_count), 
                                   index=game_count_df.index, columns=['neural net hit frequency'])
        hit_rate_df.reset_index(inplace=True)
        # print(hit_rate_df)
        
        # 保守模型在玩家手牌小于 12 时，以 1.0 的概率要牌，将该数据加入 hit_rate_df
        naive_hit_rate = []
        for i in range(4, 22):
            if i < 12:
                naive_hit_rate.append(1.0)
            else:
                naive_hit_rate.append(0.0)
        hit_rate_df['naive strategy hit frequency'] = naive_hit_rate
        
        # 输出 hit_rate_df 模型中 'neural net hit frequency', 'naive strategy hit frequency' 两列值
        data = hit_rate_df[['neural net hit frequency', 'naive strategy hit frequency']]
        # print(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x=hit_rate_df['player_total_initial']-0.2, 
               height=data['neural net hit frequency'].values, color='blue', width=0.4, label='Neural Network')
        ax.bar(x=hit_rate_df['player_total_initial']+0.2, 
               height=data['naive strategy hit frequency'].values, color='orange', width=0.4, label='Naive')
        ax.set_xlabel("Player's Hand Value", fontsize=16)
        ax.set_ylabel("Frequency of Hitting", fontsize=16)
        plt.xticks(np.arange(4, 21, 1.0))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname='./img/hit_frequency', dpi=150)
        

        # Calculate and graph the neural net's hit frequency vs. dealer card for
        # player hands in [12, 13, 14, 15, 16]
        #  -- 针对玩家点数和为 12, 13, 14, 15, 16 的数据，查看庄家第一张手牌对玩家要牌‘输’的影响
        # 筛选出玩家点数和为 12, 13, 14, 15, 16 的数据，以庄家第一张手牌和是否要牌按次序分组，统计这些局玩家输掉的值
        player_despair_df = model_df[(model_df['player_total_initial']==12) |
                                     (model_df['player_total_initial']==13) |
                                     (model_df['player_total_initial']==14) |
                                     (model_df['player_total_initial']==15) |
                                     (model_df['player_total_initial']==16)] \
                                         .groupby(by=['dealer_card_num', 'hit?']).count()['lose']
        player_despair_df2 = player_despair_df.reset_index().copy()
        print(player_despair_df, player_despair_df2)
        
        # 会有 ['hit?'] ！= 1 的情况 ❓
        hit_rate_despair = np.array(player_despair_df2[player_despair_df2['hit?']==1])[:, -1] / \
            np.array(player_despair_df2.groupby(by='dealer_card_num').sum())[:, -1]
        print(hit_rate_despair)
            
        data = pd.DataFrame(hit_rate_despair, index=player_despair_df2.groupby(by='dealer_card_num').sum().index, columns=['hit_rate'])
        print(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x=data.index, height=data['hit_rate'].values)
        ax.set_xlabel("Dealer's Card", fontsize=16)
        ax.set_ylabel("Frequency of Hitting", fontsize=16)
        plt.xticks(np.arange(2, 12, 1.0))
        
        plt.tight_layout()
        plt.savefig(fname='./img/hit_frequency2', dpi=150)
        
        # 
        print('Total hit frequency: ' + str(round(model_df[model_df['hit?']==1].shape[0] / np.sum(model_df.shape[0]), 4)))
        

# 绘制训练模型曲线图
def plot(pred_Y_train, actuals):
    # Plot ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(actuals, pred_Y_train)
    roc_auc = metrics.auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(fpr, tpr, label = ('ROC AUC = %0.3f' % roc_auc))
    
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize=16)
    
    plt.savefig(fname='./img/roc_curve_blackjack', dpi=150)
    plt.show()
    

if __name__ == '__main__':
    model_df_naive = step(0)
    model_df_random = step(1)
    presentation(1, model_df_random)
    
    model_nn, pred_Y_train, actuals = train(model_df_random)
    plot(pred_Y_train, actuals)
    model_df_nn = step(2, model_nn)
    comparison(model_df_naive, model_df_random, model_df_nn)
    print(model_df_naive, model_df_random, model_df_nn)
    presentation(2, model_df_nn)
      