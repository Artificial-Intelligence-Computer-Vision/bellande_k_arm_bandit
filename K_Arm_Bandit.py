import numpy as np
import matplotlib.pyplot as plt

class K_armed_Bandit_Problem(object):
    def __init__(self, number_bandit_problems = 500, k = 7, epsilon = 0.1, c = 1):
        self.number_bandit_problems = number_bandit_problems
        self.k = k
        self.epsilon = epsilon
        self.c = c
        self.number_of_time_step = 1000

        # Start
        self.play_k_armed_bandit()
     


    # Return Q*(a)
    def k_armed_bandit(self):
        return np.random.normal(0,1,(self.number_bandit_problems,k))
    
    

    def greedy(self, action_type):

         for i in range(self.number_of_time_step):

            action = self.action_choice(action_type = problem_action_type)
            self.action_value_reward(i, action, problem_number, q_values)

        optimal_action = self.optimal_action(q_values)
        
        return rewards, optimal_action




    def epsilon_greedy(self, action_type):
        
        for i in range(self.number_of_time_step):

            if np.random.uniform(0,1) < self.epsilon:

                action = self.action_choice(action_type = problem_action_type)
                self.action_value_reward(i, action, problem_number, q_values)
            
            else:
                action = self.action_choice(action_type = problem_action_type)
                self.action_value_reward(i, action, problem_number, q_values)
                
        optimal_action = self.optimal_action(q_values)
        return rewards, optimal_action




    def ucb(self, action_type):
        
        for i in range(self.number_of_time_step):

            action = self.action_choice(action_type = problem_action_type)
            self.action_value_reward(i, action, problem_number, q_values)

        optimal_action = self.optimal_action(q_values)
        return rewards, optimal_action



    def init_and_reset(self):

        self.q = np.zeros(self.k)
        self.rewards = np.zeros(self.number_of_time_step)
        self.action_each_step = np.zeros(self.number_of_time_step)
        self.actions_taken = np.ones(self.k)



    def action_choice(self, action_type, actions = None):

        if action_type == "greedy_action":
            action = np.argmax(self.q)
            return action
        elif action_type == "epsilon_greedy":
            action = np.random.randint(0, self.k)
            return action

        elif action_type == "ucb_action":
            action = self.ucb_action_choise()
            return action

        

    def ucb_action_choise(self, q, actions):

        action_array = np.(self.k)

        for i in range(self.k):
            action_array[i]= self.q[i] + self.c * np.sqrt(np.log(i+1) / self.actions_taken[i])

        return np.argmax(action_array)



    def action_state_reward(count, prob_num, q_values):    
        
        self.action_each_step[count]=action
        self.rewards[count] = np.random.normal(q_values[prob_num][action],1)
        self.actions_taken[action] += 1
        self.q[action] = self.q[action]+1 / self.actions_taken[action] * (self.rewards[count] - self.q[action])



    def optimal_action(self, q_values, problem_number):

        optimal_action = self.action_each_step == np.argmax(q_values[problem_number])
        return optimal_action



    def play_k_armed_bandit(self, q_values, problem_number = 0, time_step = 1000, problem_action_type = "greedy_action"):
       
        self.init_and_reset()
        
        
        for i in range(self.number_of_time_step):

            action = self.action_choice(action_type = problem_action_type)
            self.action_value_reward(i, action, problem_number, q_values)

        optimal_action = self.optimal_action(q_values)
        return rewards,optimal_action




class k_arm_bandit_q_value(object):
    def __init__(self):
        pass




class plot_colected_graphs(object):
    def __init__(self):
        pass
        
