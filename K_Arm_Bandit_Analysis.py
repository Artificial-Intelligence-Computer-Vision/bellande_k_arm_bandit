from header_import import *

if __name__ == "__main__":
    

    # Determine witch one to run 
    if sys.argv[1] == "without_baseline":


        # Array to plot
        reward_array = None
        optimal_action_array = None

        # Run k bandit
        # Greedy Method
        K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
        greedy_reward, greedy_optimal = K_armed_Bandit_Problem_obj.play(method = "greedy")
    

        # Epsilon Greedy Method
        K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
        epsilon_greedy_reward_1, epsilon_greedy_optimal_1 = K_armed_Bandit_Problem_obj.play(method = "epsilon_greedy_01")


        K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
        epsilon_greedy_reward_2, epsilon_greedy_optimal_2 = K_armed_Bandit_Problem_obj.play(method = "epsilon_greedy_001")


        # UCB Method
        K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
        ucb_reward_1, ucb_optimal_1 = K_armed_Bandit_Problem_obj.play(method = "ucb_1")

        K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
        ucb_reward_2, ucb_optimal_2 = K_armed_Bandit_Problem_obj.play(method = "ucb_2")


        # Into Array
        reward_array = [greedy_reward, epsilon_greedy_reward_1, epsilon_greedy_reward_2, ucb_reward_1, ucb_reward_2]
        optimal_action_array = [greedy_optimal, epsilon_greedy_optimal_1, epsilon_greedy_optimal_2, ucb_optimal_1, ucb_optimal_2]
        
        # plot all the
        plot_collected_graphs_obj = plot_collected_graphs(reward_array, optimal_action_array)


        
    if sys.argv[1] == "with_baseline":

        # Array to plot








