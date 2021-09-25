from header_imports import *

if __name__ == "__main__":
    

    # Determine witch one to run 
    if sys.argv[1] == "without_baseline":


        # Array to plot
        methods_array = ["greedy", "epsilon_greedy_001", "epsilon_greedy_01", "ucb_1", "ucb_2"]
        reward_array = []
        optimal_action_array = []


        for i in range(len(methods_array)):

            # Run K Bandit
            # Greedy Method, Epsilon Greedy, and UCB

            K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
            print(methods_array[i])
            reward, optimal = K_armed_Bandit_Problem_obj.play(method_type = methods_array[i])
            reward_array.append(reward)
            optimal_action_array.append(optimal)

        reward_array = np.array(reward_array)
        optimal_action_array = np.array(optimal_action_array)

    
        # plot all
        plot_collected_graphs_obj = plot_collected_graphs(reward_array, optimal_action_array, name = "none_gradient")


        
    if sys.argv[1] == "with_baseline":

        # Array to plot
        methods_array = [[0.01, 0], [0.01, 5], [0.01, 10], [0.1, 0], [0.1, 5], [0.1, 10], [0.5, 0], [0.5, 5], [0.5, 10]]
        reward_array = []
        optimal_action_array = []

        for i in range(len(methods_array)):

            K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem_Gradient(alpha = methods_array[i][0], baseline = methods_array[i][1])
            print(methods_array[i])
            reward, optimal = K_armed_Bandit_Problem_obj.play(method_type = methods_array[i], alpha_baseline = "True",  function = K_armed_Bandit_Problem_obj.Gradient_Bandit)
            reward_array.append(reward)
            optimal_action_array.append(optimal)
        
        reward_array = np.array(reward_array)
        optimal_action_array = np.array(optimal_action_array)

    
        # plot all
        plot_collected_graphs_obj = plot_collected_graphs(reward_array, optimal_action_array, name = "gradient")


        
        








