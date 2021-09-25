from header_imports import *


class K_armed_Bandit_Problem(object):
    def __init__(self, number_bandit_problems = 500, k = 7, epsilon = 0.01, c = 1):
        self.number_bandit_problems = number_bandit_problems
        self.k = k
        self.epsilon = epsilon
        self.c = c
        self.number_of_time_step = 1000


    # Return Q*(a)
    def k_armed_bandit(self):
        return np.random.normal(0,1, (self.number_bandit_problems, self.k))
    

    def greedy(self, problem_number):

        for i in range(self.number_of_time_step):
            action = self.action_choice(action_type = "greedy_action")
            self.action_value_reward(i, action, problem_number)

        optimal_action = self.optimal_action(problem_number)
        return optimal_action


    def epsilon_greedy(self, problem_number):

        for i in range(self.number_of_time_step):

            if np.random.uniform(0,1) < self.epsilon:
                action = self.action_choice(action_type = "epsilon_greedy")
                self.action_value_reward(i, action, problem_number)
            else:
                action = self.action_choice(action_type = "greedy_action")
                self.action_value_reward(i, action, problem_number)
                
        optimal_action = self.optimal_action(problem_number)
        return optimal_action


    def ucb(self, problem_number):

        for i in range(self.number_of_time_step):
            action = self.action_choice(action_type = "ucb_action")
            self.action_value_reward(i, action, problem_number)

        optimal_action = self.optimal_action(problem_number)
        return optimal_action


    def init_and_reset(self, method = "regular"):

        self.q = np.zeros(self.k)
        self.action_each_step = np.zeros(self.number_of_time_step)
        self.actions_taken = np.ones(self.k)

        if method == "gradient":
            self.actions_range = np.arange(self.k)
            self.rewards = self.action_each_step = np.zeros(self.number_of_time_step)
        else:
            self.rewards = np.zeros(self.number_of_time_step)

        self.q_values = self.k_armed_bandit()


    def action_choice(self, action_type, actions_function = None):
        
        if action_type == "greedy_action":
            action = np.argmax(self.q)
            return action

        elif action_type == "epsilon_greedy":
            action = np.random.randint(0, self.k)
            return action

        elif action_type == "ucb_action":
            action = self.ucb_action_choise()
            return action

        elif action_type == "softmax":
            action = np.random.choice(self.actions_range, p=actions_function(self.q))

            return action


    def ucb_action_choise(self):

        action_array = np.zeros(self.k)
        for i in range(self.k):
            action_array[i]= self.q[i] + self.c * np.sqrt(np.log(i+1) / self.actions_taken[i])

        return np.argmax(action_array)


    def action_value_reward(self, count, action, problem_number, baseline = "None"):

        self.action_each_step[count] = action
        self.rewards[count] = np.random.normal(self.q_values[problem_number][action],1)
        self.actions_taken[action] += 1

        if baseline == "None":
            self.q[action] = self.q[action] + 1 / self.actions_taken[action] * (self.rewards[count] - self.q[action])

        elif baseline == "True":
            self.baseline = self.baseline + 1 / (count+1) * (self.rewards[count] - self.baseline)
            
            for i in range(self.k):
                if i == action:
                    self.q[i] = self.q[i] + self.alpha * (self.rewards[i] - self.baseline) * (1 - self.softmax(self.q)[i])
                else:
                    self.q[i] = self.q[i]  - self.alpha * (self.rewards[i] - self.baseline) * (self.softmax(self.q)[i]) 




    def optimal_action(self, problem_number):

        optimal_action = self.action_each_step == np.argmax(self.q_values[problem_number])
        return optimal_action

    
    def method_identification(self, methods):
        
        if methods == "greedy":
            return methods

        elif methods == "epsilon_greedy_001":
            self.epsilon = 0.01
            methods = "epsilon_greedy"
            return methods

        elif methods == "epsilon_greedy_01":
            self.epsilon = 0.1
            methods = "epsilon_greedy"
            return methods

        elif methods == "ucb_1":
            self.c = 1
            methods = "ucb"
            return methods

        elif methods == "ucb_2":
            self.c = 2
            methods = "ucb"
            return methods


    # Methods  -- greedy, epsilon_greedy, ucb
    def play_k_armed_bandit(self, problem_number = 0, methods = "greedy"):

        # Resets Values for each iteration
        self.init_and_reset()
        methods = self.method_identification(methods)

        if methods == "greedy":
            optimal_action = self.greedy(problem_number)
            return self.rewards, optimal_action

        elif methods == "epsilon_greedy":
            optimal_action = self.epsilon_greedy(problem_number)
            return self.rewards, optimal_action

        elif methods  == "ucb":
            optimal_action = self.ucb(problem_number)
            return self.rewards, optimal_action
        

    # Where it is to play the k armed bandit
    def play(self, method_type, alpha_baseline = "False", function=None):
        
        reward = np.zeros((self.number_bandit_problems, self.number_of_time_step))
        optimal = np.zeros((self.number_bandit_problems, self.number_of_time_step))
        
        if alpha_baseline == "False":
            for i in range(self.number_bandit_problems):
                reward[i], optimal[i] = self.play_k_armed_bandit(problem_number = i, methods = method_type)
            return reward, optimal

        elif alpha_baseline == "True":
            for i in range(self.number_bandit_problems):
                reward[i], optimal[i] = function(problem_number = i)
            return reward, optimal


class plot_collected_graphs(object):
    def __init__(self, reward_array, optimal_action_array, name):
        
        # Path
        self.path = "graph_and_chart/"
        self.pdf_type = "regular_pdf/"
        
        self.type_name = name
        self.rewards_array = reward_array
        self.optimal_action_array = optimal_action_array

        self.true_path = self.path + self.pdf_type
        
        if self.type_name == "none_gradient":

            self.plot_graphs_methods(array_first = "reward")
            self.plot_graphs_methods(array_first = "optimal_action")

            self.plot_graph_greedy(array_first = "reward")
            self.plot_graph_greedy(array_first = "optimal_action")

            self.plot_graph_epsilon_greedy(array_first = "reward", epsilon_name = "0.01")
            self.plot_graph_epsilon_greedy(array_first = "optimal_action", epsilon_name = "0.01")


            self.plot_graph_epsilon_greedy(array_first = "reward", epsilon_name = "0.1")
            self.plot_graph_epsilon_greedy(array_first = "optimal_action", epsilon_name = "0.1")

            self.plot_graph_ucb(array_first = "reward", ucb_name = "1")
            self.plot_graph_ucb(array_first = "optimal_action", ucb_name = "1")

            self.plot_graph_ucb(array_first = "reward", ucb_name = "2")
            self.plot_graph_ucb(array_first = "optimal_action", ucb_name = "2")

        elif self.type_name == "gradient":
            
            self.plot_graph_method_gradient(array_first = "reward")
            self.plot_graph_method_gradient(array_first = "optimal_action")


    def plot_graph_method_gradient(self, array_first):
        
        if array_first == "reward":

            plt.figure(figsize=(40,16))
            plt.title('Average Reward vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Reward', fontsize=16)
            plt.plot(self.rewards_array[0].mean(axis=0), label="Reward Alpha 0.01 & Baseline 0")
            plt.plot(self.rewards_array[1].mean(axis=0), label="Reward Alpha 0.01 & Baseline 5")
            plt.plot(self.rewards_array[2].mean(axis=0), label="Reward Alpha 0.01 & Baseline 10")
            plt.plot(self.rewards_array[3].mean(axis=0), label="Reward Alpha 0.1  & Baseline 0")
            plt.plot(self.rewards_array[4].mean(axis=0), label="Reward Alpha 0.1 & Baseline 5")
            plt.plot(self.rewards_array[5].mean(axis=0), label="Reward Alpha 0.1 & Baseline 10")
            plt.plot(self.rewards_array[6].mean(axis=0), label="Reward Alpha 0.5 & Baseline 0")
            plt.plot(self.rewards_array[7].mean(axis=0), label="Reward Alpha 0.5 & Baseline 5")
            plt.plot(self.rewards_array[8].mean(axis=0), label="Reward Alpha 0.5 & Baseline 10")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "_reward_methods_compare.png"), dpi =500)


        elif array_first == "optimal_action":

            plt.figure(figsize=(40,16))
            plt.title('Optimal Action vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Optimal', fontsize=16)
            plt.plot(self.optimal_action_array[0].mean(axis=0)*100, label="Optimal Alpha 0.01 & Baseline 0")
            plt.plot(self.optimal_action_array[1].mean(axis=0)*100, label="Optimal Alpha 0.01 & Baseline 5")
            plt.plot(self.optimal_action_array[2].mean(axis=0)*100, label="Optimal Alpha 0.01 & Baseline 10")
            plt.plot(self.optimal_action_array[3].mean(axis=0)*100, label="Optimal Alpha 0.1  & Baseline 0")
            plt.plot(self.optimal_action_array[4].mean(axis=0)*100, label="Optimal Alpha 0.1 & Baseline 5")
            plt.plot(self.optimal_action_array[5].mean(axis=0)*100, label="Optimal Alpha 0.1 & Baseline 10")
            plt.plot(self.optimal_action_array[6].mean(axis=0)*100, label="Optimal Alpha 0.5 & Baseline 0")
            plt.plot(self.optimal_action_array[7].mean(axis=0)*100, label="Optimal Alpha 0.5 & Baseline 5")
            plt.plot(self.optimal_action_array[8].mean(axis=0)*100, label="Optimal Alpha 0.5 & Baseline 10")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "_optimal_methods_compare.png"), dpi =500)


    def plot_graphs_methods(self, array_first):

        if array_first == "reward":

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
            plt.savefig((str(self.true_path) + self.type_name + "_reward_methods_compare.png"), dpi =500)



        elif array_first == "optimal_action":

            plt.figure(figsize=(40,16))
            plt.title('Optimal Action vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Optimal Action in %', fontsize=16)
            plt.plot(self.optimal_action_array[0].mean(axis=0)*100, label="Optimal Greedy Method")
            plt.plot(self.optimal_action_array[1].mean(axis=0)*100, label="Optimal Epsilon Greedy 0.01 Method")
            plt.plot(self.optimal_action_array[2].mean(axis=0)*100, label="Optimal Epsilon Greedy 0.1 Method")
            plt.plot(self.optimal_action_array[3].mean(axis=0)*100, label="Optimal UCB 1 Method")
            plt.plot(self.optimal_action_array[4].mean(axis=0)*100, label="Optimal UCB 2 Method")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "_optimal_methods_compare.png"), dpi =500)


    def plot_graph_greedy(self, array_first):

        if array_first == "rewards":

            plt.figure(figsize=(40,16))
            plt.title('Average Reward vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Reward', fontsize=16)
            plt.plot(self.rewards_array[0].mean(axis=0), label="Reward Greddy Method")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "_reward_greedy_method.png"), dpi =500)


        elif array_first == "optimal_action":

            plt.figure(figsize=(40,16))
            plt.title('Optimal Action vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Optimal Action in %', fontsize=16)
            plt.plot(self.optimal_action_array[0].mean(axis=0)*100, label="Optimal Greddy Method")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "_optimal_greedy_method.png"), dpi =500)


    def plot_graph_epsilon_greedy(self, array_first, epsilon_name):
       

        if array_first == "rewards":

            if epsilon_name == "0.01":
                name = "0.01"
                reward = self.rewards_array[1]
            elif epsilon_name == "0.1":
                name = "0.1"
                reward = self.rewards_array[2]

            plt.figure(figsize=(40,16))
            plt.title('Average Reward vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Reward', fontsize=16)
            plt.plot(reward.mean(axis=0), label="Reward Epsilon Greedy Method")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "reward_greedy_method_" + name + "_.png"), dpi =500)


        elif array_first == "optimal_action":

            if epsilon_name == "0.01":
                name = "0.01"
                optimal = self.optimal_action_array[1]
            elif epsilon_name == "0.1":
                name = "0.1"
                optimal = self.optimal_action_array[2]

            plt.figure(figsize=(40,16))
            plt.title('Optimal Action vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Optimal Action in %', fontsize=16)
            plt.plot(optimal.mean(axis=0)*100, label="Optimal Greedy Method")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "optimal_greedy_method_" + name + "_.png"), dpi =500)


    def plot_graph_ucb(self, array_first, ucb_name):
       
        if array_first == "reward":

            if ucb_name == "1":
                name = "1"
                reward = self.rewards_array[3]
            elif ucb_name == "2":
                name = "2"
                reward = self.rewards_array[4]

            plt.figure(figsize=(40,16))
            plt.title('Average Reward vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Reward', fontsize=16)
            plt.plot(reward.mean(axis=0), label="Reward UCB Method")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "reward_ucb_method_" + name + "_.png"), dpi =500)


        elif array_first == "optimal_action":

            if ucb_name == "1":
                name = "1"
                optimal = self.optimal_action_array[3]
            elif ucb_name == "2":
                name = "2"
                optimal = self.optimal_action_array[4]

            plt.figure(figsize=(40,16))
            plt.title('Optimal Action vs Time steps')
            plt.xlabel('Time_steps', fontsize=18)
            plt.ylabel('Optimal Action in %', fontsize=16)
            plt.plot(optimal.mean(axis=0)*100, label="Optimal UCB Method")
            plt.legend()
            plt.savefig((str(self.true_path) + self.type_name + "optimal_ucb_method_" + name + "_.png"), dpi =500)


# Gradient Bandit with baseline
class K_armed_Bandit_Problem_Gradient(K_armed_Bandit_Problem):
    def __init__(self, alpha = 0.01, baseline = 0):
        super().__init__()

        self.alpha = alpha
        self.baseline = baseline


    def softmax(self, q):
        return np.exp(q) / np.exp(q).sum()

    
    def Gradient_Bandit(self, problem_number):
        
        self.init_and_reset(method = "gradient")

        for i in range(self.number_of_time_step):
            action = self.action_choice(action_type = "softmax", actions_function = self.softmax)
            self.action_value_reward(i, action, problem_number, baseline = "True")

        optimal_action = self.optimal_action(problem_number)
        return self.rewards, optimal_action






        



                
        
