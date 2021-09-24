from header_import import *

if __name__ == "__main__":
    

    # Determine witch one to run 
    if sys.argv[1] == "without_baseline":


        # Array to plot
        reward_array = None
        optimal_action_array = None
        
        methods_array = ["greedy", "epsilon_greedy_001", "epsilon_greedy_01", "ucb_1", "ucb_2"]
        
        
        for i in range(5):

            # Run K Bandit
            # Greedy Method, Epsilon Greedy, and UCB
            K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
            reward_array[i], optimal_action_array[i] = K_armed_Bandit_Problem_obj.play(method = methods_array[i])
    
        # plot all
        plot_collected_graphs_obj = plot_collected_graphs(reward_array, optimal_action_array)


        
    if sys.argv[1] == "with_baseline":

        # Array to plot

        reward_array = None
        optimal_action_array = None
        
        alpha = [0.01, 0.1, 0.5]
        baseline = [0, 5, 10]

        for i in it.product(alpha, baseline):

            a , b = i
            K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem_Gradient()
            reward_array[i], optimal_action_array[i] = K_armed_Bandit_Problem_obj.play(method = methods_array[i], function)
        
        
        








