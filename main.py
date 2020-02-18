# %%
import generating
import useful_utils
import params

useful_utils.print_section_seperator("Experiment Parameters")
print("Experiment scenario:", params.SCENARIO_Name)
print(
    "n = %d, m = %d, s = %d."
    % (params.data_prop_dict["n"], params.data_prop_dict["m"], params.data_prop_dict["s"])
)
print(
    "N-Train = %d, N-Test = %d."
    % (params.data_prop_dict["N_TRAIN"], params.data_prop_dict["N_TEST"])
)
# %% Create Data
# Data properties
data_generating = generating.DataPropGeneration(**params.data_prop_dict)
# Create simulated data
training_data = data_generating.create_data(params.training_data_dict)
# Create simulated data also for trainig Oracle LISTA
training_data.create_data_oracle_lista()

# %% Train the different solvers
useful_utils.print_section_seperator("Training")
print(f"Number of unfoldings (T): start-{params.tstart}, end-{params.tend}, step-{params.tstep}.")
T_opt = range(params.tstart, params.tend, params.tstep)
err_list = []
for i in range(len(T_opt)):
    training_data.T = T_opt[i]
    err_dict = training_data.training_procedure()
    # Print errors
    err_msg = f"T {training_data.T}"
    for err_i in err_dict:
        err_msg += ", %s %.8f" % (err_i, err_dict[err_i])
    useful_utils.print_section_seperator(err_msg, subsec=True)
    err_list.append(err_dict)
# %% Plot the figures
training_data.plot_figure(T_opt=T_opt, err_list=err_list, design_dict=params.design_dict)
