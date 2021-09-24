from header_import import *


class K_armed_Bandit_Problem(object):
    def __init__(self, number_bandit_problems = 500, k = 7, epsilon = 0.1, c = 1):
        self.number_bandit_problems = number_bandit_problems
        self.k = k
        self.epsilon = epsilon
        self.c = c
        self.number_of_time_step = 1000

     


    # Return Q*(a)
    def k_armed_bandit(self):
        return np.random.normal(0,1,(self.number_bandit_problems, self.k))
    
    

    def greedy(self, q_values):

         for i in range(self.number_of_time_step):

            action = self.action_choice(action_type = "greedy_action")
            self.action_value_reward(i, action, problem_number, q_values)

        optimal_action = self.optimal_action(q_values)
        
        return optimal_action




    def epsilon_greedy(self, q_values):
        
        for i in range(self.number_of_time_step):

            if np.random.uniform(0,1) < self.epsilon:

                action = self.action_choice(action_type = "epsilon_greedy")
                self.action_value_reward(i, action, problem_number, q_values)
            
            else:
                action = self.action_choice(action_type = "greedy_action")
                self.action_value_reward(i, action, problem_number, q_values)
                
        optimal_action = self.optimal_action(q_values)
        return optimal_action




    def ucb(self, q_values):
        
        for i in range(self.number_of_time_step):

            action = self.action_choice(action_type = "ucb_action")
            self.action_value_reward(i, action, problem_number, q_values)

        optimal_action = self.optimal_action(q_values)
        return optimal_action



    def init_and_reset(self):

        self.q = np.zeros(self.k)
        self.rewards = np.zeros(self.number_of_time_step)
        self.action_each_step = np.zeros(self.number_of_time_step)
        self.actions_taken = np.ones(self.k)
        self.q_value = self.k_armed_bandit()



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



    # Methods  -- greedy, epsilon_greedy, ucb
    def play_k_armed_bandit(self, q_values, problem_number = 0, methods = "greedy"):
       
        # Resets Values for each iteration
        self.init_and_reset()
        
        if methods == "greedy":
            optimal_action= self.greedy(q_values)
            return self.rewards, optimal_action

        elif methods == "epsilon_greedy":
            optimal_action = self.epsilon_greedy(q_values)
            return self.rewards optimal_action

        elif methods  == "ucb":
            optimal_action = self.ucb(q_values)
            return self.rewards, optimal_action
        




class plot_colected_graphs(object):
    def __init__(self, reward_array, optimal_action_array):
        
        # Path
        self.path = "/graph_and_charts/"
        self.pdf_type = "/regular_pdf"

        self.reward_array = reward_array
        self.optimal_action_array = optimal_action_array


        self.true_path = self.path + self.pdf_type

        self.plot_graphs_methods()
        

        # Print each seperately







    def plot_graphs_methods(self):

        plt.figure(figsize=(40,16))
        plt.title('Average Reward vs Time steps')
        plt.xlabel('Time_steps', fontsize=18)
        plt.ylabel('Reward', fontsize=16)
        plt.plot(self.rewards_array[0].mean(axis=0), label="Reward Greedy Method")
        plt.plot(self.rewards_array[1].mean(axis=0), label="Reward Epsilon Greedy 0.01 Method")
        plt.plot(self.rewards_array[2].mean(axis=0), label="Reward Epsilon Greedy 0.1 Method")
        plt.plot(self.rewards_array[3].mean(axis=0), label="Reward UCB 1 Method")
        plt.plot(self.rewards_array[4].mean(axis=0), label="Reward UCB 2 Method")
        plt.legend()
        plt.savefig((str(self.true_path) + "methods_compare.png"), dpi =500)



    def plot_graph_greedy(self):

        plt.figure(figsize=(40,16))
        plt.title('Average Reward vs Time steps')
        plt.xlabel('Time_steps', fontsize=18)
        plt.ylabel('Reward', fontsize=16)
        plt.plot(self.rewards_array[0].mean(axis=0), label="Reward Greddy Method")
        plt.legend()
        plt.savefig((str(self.true_path) + "greedy_method.png"), dpi =500)



    def plot_graph_epsilon_greedy(self):
        
        if epsilon_name == "0.01":
            name = "0.01"
            reward = self.reward_array[1]
        elif epsilon_name == "0.1":
            name = "0.1"
            reward = self.reward_array[2]


        plt.figure(figsize=(40,16))
        plt.title('Average Reward vs Time steps')
        plt.xlabel('Time_steps', fontsize=18)
        plt.ylabel('Reward', fontsize=16)
        plt.plot(reward.mean(axis=0), label="Reward Epsilon Greedy Method")
        plt.legend()
        plt.savefig((str(self.true_path) + "greedy_method_" + name + "_.png"), dpi =500)




    def plot_graph_ucb(self):
        
        if ucb_name == "1":
            name = "1"
            reward = self.reward_array[3]
        elif ucb_name == "2":
            name = "2"
            reward = self.reward_array[4]



        plt.figure(figsize=(40,16))
        plt.title('Average Reward vs Time steps')
        plt.xlabel('Time_steps', fontsize=18)
        plt.ylabel('Reward', fontsize=16)
        plt.plot(reward.mean(axis=0), label="Reward UCB Method")
        plt.legend()
        plt.savefig((str(self.true_path) + "ucb_method_" + name + "_.png"), dpi =500)



        
