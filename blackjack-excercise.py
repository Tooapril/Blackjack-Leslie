import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    return list(set([int(s) for s in np.sum(sum_array, axis=1) if s<=21])) #⭐️ 将所有 'A' 能组成总和不超过 21 的值返回
    
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
    
 
def play(is_random):    
    stacks = 50000 # 牌局数目
    players = 1 # 玩家数目
    num_decks = 1 # 牌副数目

    card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    dealer_card_feature = []
    player_card_feature = []
    player_results = []

    for stack in range(stacks):
        blackjack = set(['A', 10])
        dealer_cards = make_decks(num_decks, card_types) # 根据给定牌副数洗牌
        while len(dealer_cards) > 20: # 牌盒里的牌不大于 20 张就没必要继续用这副牌进行游戏 -- 规则⭐️
            
            current_player_results = np.zeros((1, players))
            
            dealer_hand = []
            player_hands = [[] for player in range(players)]
            
            # Deal FIRST card
            for player, hand in enumerate(player_hands): # 先给所有玩家发第一张牌
                player_hands[player].append(dealer_cards.pop(0)) # 将洗好的牌分别发给玩家
            dealer_hand.append(dealer_cards.pop(0)) # 再给庄家发第一张牌
            # Deal SECOND card
            for player, hand in enumerate(player_hands): # 先给所有玩家发第二张牌
                player_hands[player].append(dealer_cards.pop(0)) # 接着刚刚洗好的牌继续发牌
            dealer_hand.append(dealer_cards.pop(0)) # 再给庄家发第二张牌
            
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
                        if(is_random):
                            # Hit randomly, check for busts -- 在玩家当前手牌不是二十一点时，以随机值大于 0.5 的方式决定是否拿牌
                            while (random.random() >= 0.5) and (total_up(player_hands[player]) != 21):
                                player_hands[player].append(dealer_cards.pop(0))
                                if total_up(player_hands[player]) > 21: # 拿完牌后再次确定是否爆牌，爆牌则结果为 -1
                                    current_player_results[0, player] = -1
                                    break
                        else:
                            # Hit only when we know we will not bust -- 在玩家当前手牌点数不超过 11 时，才决定拿牌
                            while total_up(player_hands[player]) <= 11:
                                player_hands[player].append(dealer_cards.pop(0))
                                if total_up(player_hands[player]) > 21: # 拿完牌后再次确定是否爆牌，爆牌则结果为 -1
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
                        
                    # print('player: ' + str(total_up(player_hands[player])),
                    #         'dealer: ' + str(total_up(dealer_hand)),
                    #         'result: ' + str(current_player_results)
                    #         )
            
            # Track features
            dealer_card_feature.append(dealer_hand[0]) # 将庄家的第一张牌存入新的 list
            player_card_feature.append(player_hands) # 将每个玩家当前手牌存入新的 list
            player_results.append(list(current_player_results[0])) # 将各玩家的输赢结果存入新的 list
                    
    model_df = pd.DataFrame() # 构造数据集
    model_df['dealer_card'] = dealer_card_feature # 所有游戏庄家的第一张牌
    model_df['player_total'] = [total_up(i[0]) for i in player_card_feature] # 所有游戏第一个玩家的点数和（第一个玩家 -- 作为数据分析对象❗️）
    model_df['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature] # 所有游戏第一个玩家前两张牌的点数和（第一个玩家 -- 作为数据分析对象❗️）
    model_df['Y'] = [i[0] for i in player_results] # 所有游戏第一个玩家输赢结果（第一个玩家 -- 作为数据分析对象❗️）

    lose = []
    for i in model_df['Y']:
        if i == -1: # 玩家输，lose 列表追加一个 1，e.g. [1, 1, ...]
            lose.append(1)
        else: # 玩家平局或赢，lose 列表追加一个 0，e.g. [0, 0, ...]
            lose.append(0)
    model_df['lose'] = lose

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
    
    # 统计玩家一的所有输、赢、平的次数
    # -1.0    199610
    #  1.0     99685
    #  0.0     13289
    # Name: 0, dtype: int64 
    # 312584
    count = pd.DataFrame(player_results)[0].value_counts()
    print(count, sum(count))

    return model_df


def save_picture(model_df, model_df_smart, position):
    data = pd.DataFrame()
    
    if position == 'dealer':
        # 两种模型对比：查看庄家第一张牌牌值对玩家“不输”的影响
        # 保守模型
        data_smart = 1 - (model_df_smart.groupby(by='dealer_card_num').sum()['lose'] / 
                          model_df_smart.groupby(by='dealer_card_num').count()['lose'])
        # 随机模型
        data_random = 1 - (model_df.groupby(by='dealer_card_num').sum()['lose'] / 
                           model_df.groupby(by='dealer_card_num').count()['lose'])
        
        data['smart'] = data_smart
        data['random'] = data_random
    elif position == 'player':
        # 两种模型对比：查看玩家前两张初始牌值对玩家“不输”的影响
        # 保守模型
        data_smart = 1 - (model_df_smart.groupby(by='player_total_initial').sum()['lose'] / 
                          model_df_smart.groupby(by='player_total_initial').count()['lose'])
        # 随机模型
        data_random = 1 - (model_df.groupby(by='player_total_initial').sum()['lose'] / 
                           model_df.groupby(by='player_total_initial').count()['lose'])

        data['smart'] = data_smart[:-1]
        data['random'] = data_random[:-1]

    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x=data.index-0.2, height=data['smart'].values, color='blue', width=0.4, label='Smart')
    ax.bar(x=data.index+0.2, height=data['random'].values, color='red', width=0.4, label='Coin Flip')
    ax.set_xlabel(position.capitalize() + "'s Card", fontsize=16) # capitalize() -- 将字符串首字母大写
    ax.set_ylabel('Probability of Tie or Win', fontsize=16)
    plt.xticks(np.arange(2, 12, 1.0))

    plt.legend()
    plt.tight_layout()
    plt.savefig(fname= './img/' + position + '_card_probs_smart', dpi=150)


if __name__ == '__main__':
    model_df = play(True) # 随机模型
    model_df_smart = play(False) # 保守模型

    print(model_df)
    print(model_df_smart)
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
    print(model_df.groupby(by='has_ace').sum()['lose'] / 
          model_df.groupby(by='has_ace').count()['lose'])

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

    # 去掉玩家初始 21 点的情况后，玩家手牌与庄家第一张牌对玩家“不输”的影响
    model_tmp = model_df[model_df['player_total_initial'] != 21] # 去掉玩家一初始手牌点数为 21 的数据
    pivot_data2 = model_tmp[model_tmp['player_total'] < 21]
    losses_pivot2 = pd.pivot_table(pivot_data2, values='lose', index=['dealer_card_num'], 
                                  columns=['player_total'], aggfunc=np.sum)
    games_pivot2 = pd.pivot_table(pivot_data2, values='lose', index=['dealer_card_num'], 
                                 columns=['player_total'], aggfunc='count')
    heat_data2 = 1 - losses_pivot2.sort_index(ascending=False) / games_pivot2.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heat_data2, square=False, cmap="PiYG")
    ax.set_xlabel("Player's Hand Value", fontsize=16)
    ax.set_ylabel("Dealer's Card", fontsize=16)

    plt.savefig(fname='./img/heat_map_random2', dpi=150)


    save_picture(model_df, model_df_smart, 'dealer')
    save_picture(model_df, model_df_smart, 'player')
    