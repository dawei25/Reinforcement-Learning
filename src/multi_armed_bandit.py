import numpy as np
import src.random


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains the MultiArmedBandit on an OpenAI Gymnasium environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size, use
        1/N, where N is the number of times the current action has been
        performed. (This is the version of Bandits we saw in lecture before
        we introduced alpha). Use an epsilon-greedy approach to pick actions.

        See (https://gymnasium.farama.org/) for examples of how to use the OpenAI
        Gymnasium Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "terminated or truncated" returned
            from env.step() is True.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length `num_bins`.
            Let s = int(np.ceil(steps / `num_bins`)), then rewards[0] should
            contain the average reward over the first s steps, rewards[1]
            should contain the average reward over the next s steps, etc.
            Please note that: The total number of steps will not always divide evenly by the 
            number of bins. This means the last group of steps may be smaller than the rest of 
            the groups. In this case, we can't divide by s to find the average reward per step 
            because we have less than s steps remaining for the last group.
        """

        # set up Q function, rewards
        n_actions, n_states = env.action_space.n, env.observation_space.n
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)
        avg_rewards = np.zeros([num_bins])
        all_rewards = []

        # reset environment before your first action
        env.reset()

        for t in range(steps):
            # Decide whether to explore or exploit
            if src.random.rand() < self.epsilon:
                # Explore: choose action at random
                # Even for random choice, use src.random
                action = src.random.randint(0, n_actions)
            else:
                # Exploit: choose the action with the highest Q-value
                # Randomly break ties among the top Q-values
                max_q = np.max(self.Q)
                best_actions = np.where(self.Q == max_q)[0]
                action = src.random.choice(best_actions)

            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            all_rewards.append(reward)

            # Update counts and Q-values
            self.N[action] += 1
            alpha = 1.0 / self.N[action]
            self.Q[action] += alpha * (reward - self.Q[action])

            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs

        # Compute averaged rewards in bins
        # Let s = int(np.ceil(steps / num_bins))
        s = int(np.ceil(steps / num_bins))
        avg_rewards = np.zeros(num_bins)
        for i in range(num_bins):
            start = i * s
            end = min((i + 1) * s, steps)
            if end > start:  # To handle the last bin if it's smaller
                avg_rewards[i] = np.mean(all_rewards[start:end])
            else:
                avg_rewards[i] = 0.0  # No steps in this bin, though unlikely

        # Replicate Q-values for each state
        state_action_values = np.tile(self.Q, (n_states, 1))

        return state_action_values, avg_rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `terminated or truncated=True`.
          - When choosing to exploit the best action, do not use np.argmax: it
            will deterministically break ties by choosing the lowest index of
            among the tied values. Instead, please *randomly choose* one of
            those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        # reset environment before your first action
        env.reset()

        # Prediction involves running one episode to completion.
        states = []
        actions = []
        rewards = []

        # Reset environment before the first action
        obs, info = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Exploit only: choose best action from Q(s,a) for current obs
            q_values = state_action_values[obs]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = src.random.choice(best_actions)

            # Take the step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Record transitions
            states.append(next_obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

        return np.array(states), np.array(actions), np.array(rewards)
