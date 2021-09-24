from header_import import *

if __name__ == "__main__":
    
    # Array for
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
    epsilon_greedy_reward_1, epsilon_greedy_optimal_1 = K_armed_Bandit_Problem_obj.play(method = "ucb_1")


    K_armed_Bandit_Problem_obj = K_armed_Bandit_Problem()
    epsilon_greedy_reward_2, epsilon_greedy_optimal_2 = K_armed_Bandit_Problem_obj.play(method = "ucb_2")





    # 

