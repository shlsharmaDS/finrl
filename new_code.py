import json
import sys

def create_env(time_window, df_portfolio_train, df_portfolio_test, tics_in_portfolio):
    environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=time_window,
        features=["close", "high", "low"],
        time_column="day",
        normalize_df=None, # dataframe is already normalized
        tics_in_portfolio=tics_in_portfolio
    )

    environment_test = PortfolioOptimizationEnv(
            df_portfolio_test,
            initial_amount=100000,
            comission_fee_pct=0.0025,
            time_window=time_window,
            features=["close", "high", "low"],
            time_column="day",
            normalize_df=None, # dataframe is already normalized
            tics_in_portfolio=tics_in_portfolio
        )
    return environment_train, environment_test

def create_model(time_window, soft_temp, GPM, new_edge_index, new_edge_type, nodes_to_select, device, environment_train):
    # set PolicyGradient parameters
    model_kwargs = {
        "lr": 0.01,
        "policy": GPM,
    }

    # here, we can set GPM's parameters
    policy_kwargs = {
        "edge_index": new_edge_index,
        "edge_type": new_edge_type,
        "nodes_to_select": nodes_to_select,
        "softmax_temperature":soft_temp,
        "device":device,
        # "k_short":6,
        # "k_medium":25,
        "time_window":time_window,
    }

    model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
    return model

def test_model(tw, st, device, environment_train, environment_test, model, new_edge_index, new_edge_type, nodes_to_select):
    GPM_results = {
        "train": environment_train._asset_memory["final"],
        "test": {},
    }

    # instantiate an architecture with the same arguments used in training
    # and load with load_state_dict.
    policy = GPM(new_edge_index, new_edge_type, nodes_to_select,
        softmax_temperature=st,
        device=device,
        # k_short=6,
        # k_medium=25,
        time_window=tw)
    policy.load_state_dict(torch.load("policy_GPM_tw_"+str(tw)+"_st_"+str(st)+".pt", map_location=torch.device('cpu')))

    # testing
    DRLAgent.DRL_validation(model, environment_test, policy=policy)
    GPM_results["test"] = environment_test._asset_memory["final"]
    file_name = 'data_tw_'+str(tw)+'_st_'+str(st)+'.json'
    with open(file_name, 'w') as json_file:
        json.dump(GPM_results, json_file, indent=4)
    print('dict file written')

def git_and_write_console_output(tw, st):
    # Save the captured output to a text file
    file_name = 'console_output_tw_'+str(tw)+'_st_'+str(st)+'.txt'
    # Open a file in write mode
    with open(file_name, 'w') as f:
        # Save the current stdout
        original_stdout = sys.stdout
        try:
            # Redirect stdout to the file
            sys.stdout = f
            # Your code here
            print("This is the console output")
            for i in range(5):
                print(f"Line {i}")
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout



time_window = [50, 70, 90]
soft_temp = [3000, 4000, 5000]

for tw in time_window:
    for st in soft_temp:
        print(f'Running for Time Window {tw} and Soft Temperature {st}')
        if tw==50 and st==3000:
            pass
        elif tw==50 and st==4000:
            pass
        else:
            environment_train, environment_test = create_env(tw,df_portfolio_train, df_portfolio_test, tics_in_portfolio)
            model = create_model(tw, st, GPM, new_edge_index, new_edge_type, nodes_to_select, device, environment_train)

            DRLAgent.train_model(model, episodes=16)
            torch.save(model.train_policy.state_dict(), "policy_GPM_tw_"+str(tw)+"_st_"+str(st)+".pt")
            #test_model(tw, st, device, environment_train, environment_test, model, new_edge_index, new_edge_type, nodes_to_select)
            git_and_write_console_output(tw, st)
