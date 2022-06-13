"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 6/5/2022 10:32 AM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : jun_test.py
"""
class Yuan(object):
    a = 1
    b = "32"


class Test(object):
    def __init__(self, env_id, entry_point=None):
        ''' Initilize

        Args:
            env_id (string): The name of the environent
            entry_point (string): A string the indicates the location of the envronment class
        '''
        self.env_id = env_id
        # mod_name, class_name = entry_point.split(':')
        test_yuan = Yuan()
        self._entry_point = getattr(Yuan, 'a')

    def make(self, a):
        ''' Instantiates an instance of the environment

        Returns:
            env (Env): An instance of the environemnt
            config (dict): A dictionary of the environment settings
        '''
        env = self._entry_point(a)
        return env

if __name__ == '__main__':
    test = Test(2)
    print(test.make('a'))