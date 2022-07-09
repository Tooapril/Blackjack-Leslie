''' Register new environments
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import register, make

register(
    env_id='blackjack',
    entry_point='rlcard.envs.blackjack:BlackjackEnv',
)

register(
    env_id='doudizhu',
    entry_point='rlcard.envs.doudizhu:DoudizhuEnv',
)

register(
    env_id='uno',
    entry_point='rlcard.envs.uno:UnoEnv',
)