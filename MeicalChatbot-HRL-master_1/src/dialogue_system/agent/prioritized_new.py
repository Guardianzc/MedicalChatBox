import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Simple replay buffer to store and sample transition experiences
    """
    def __init__(self, size):
        """
        Constructor function
        args:
            size (int) : Maximum size of replay buffer
        """
        self._maxsize = size
        self._storage = deque(maxlen=size)

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add transition data to the replay buffer
        args:
            state : Current state
            action : Action taken
            reward (float) : Received reward
            next_state : Next state
            done (bool) : Episode done
        """
        data = (state, action, reward, next_state, done)
        self._storage.append(data)

    def _encode_sample(self, idxes):
        """
        Sample data from given indexes
        args:
            idxes (list/np.array) : List with indexes of data to sample
        returns:
            np.array, np.array, np.array, np.array, np.array : Sampled states, actions, rewards, next_states and dones
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxes:
            obs_t, action, reward, obs_tp1, done = self._storage[i]
            states.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        """
        Sample data from the replay buffer
        args:
            batch_size (int) : Maximum batch size to sample
        returns:
            tuple of 5 lists : Sampled batch of transitions
        """
        batch_size = min(len(self), batch_size)
        idxes = np.random.randint(0, len(self), size=batch_size)
        return self._encode_sample(idxes)

    def clear(self):
        """
        Clear the contents of replay buffer
        """
        self._storage.clear()


class PrioritizedReplayBuffer(object):

    def __init__(self, buffer_size):
        self._priorities = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self._priorities)

    def add(self, state, action, reward, next_state, episode_over, error):
        self._priorities.append((state, action, reward, next_state, episode_over, error ))

    def sample(self, batch_size, priority_scale=1.0):
        batch_size = min(len(self._priorities), batch_size)
        batch_probs = self.get_probabilities(priority_scale)
        #print(len(self._priorities),len(batch_probs))
        #print(batch_probs)
        batch_indices = np.random.choice(range(len(self._priorities)), size=batch_size, p=batch_probs)
        #batch_importance = self.get_importance(batch_probs[batch_indices])
        batch = [self._priorities[x][:5] for x in batch_indices]

        return batch

    def get_probabilities(self, priority_scale):
        td_errors = np.array([abs(x[5]) for x in self._priorities])
        #print(td_errors)
        scaled_priorities = td_errors ** priority_scale
        batch_probabilities = scaled_priorities / sum(scaled_priorities)
        return batch_probabilities

    def get_importance(self, probabilities):
        importance = 1 / (len(self._priorities) * probabilities+0.001)  # TODO: The change here might create problem
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self._priorities[i] = abs(e) + offset