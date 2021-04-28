# Modified by Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class Memory(ABC):
    '''Abstract Memory class to define the API methods'''

    def __init__(self, paramter):
        '''
        @param {*} body is the unit that stores its experience in this memory. Each body has a distinct memory.
        '''
        self.parameter = paramter
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']

    @abstractmethod
    def reset(self):
        '''Method to fully reset the memory storage and related variables'''
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        '''Implement memory update given the full info from the latest timestep. NOTE: guard for np.nan reward and done when individual env resets.'''
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        raise NotImplementedError
