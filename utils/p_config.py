import configparser as cp  # import the ConfigParser module
import sys
sys.path.append('..')

# Create a ConfigParser object and read in the 'param.conf' file
param_config = cp.ConfigParser()
param_config.read('config/param.conf')

# Create a ConfigParser object and read in the 'graph.conf' file
graph_config = cp.ConfigParser()
graph_config.read('config/graph.conf')

# Get the number of nodes in the complete graph
n0 = graph_config["COMPLETEPARAM"]["n0"]
# Get the number of rounds
T = param_config["ALGOCONFIG"]["t"]
# Get the number of steps in the Frank-Wolfe algorithm
L = param_config["ALGOCONFIG"]["l"]
# Get the number of trials
num_trials = param_config["EXPERIMENTCONFIG"]["num_trials"]
# Get the batch size
batch_size = param_config["ALGOCONFIG"]["batch_size"]

def checkInt(str):
    """
    Check if a given string represents an integer value.
    Args:
        str: The string to check.

    Returns:
        True if the string represents an integer, False otherwise.
    """
    if str[0] in ('-', '+'):
        return str[1:].isdigit()
    return str.isdigit()

def grid_test(size, param):
    """
    Test if the size of the grid graph and the parameters match.

    Args:
        size: The size of the graph.
        param: The parameters of the graph, a list of two integers representing the number of rows and columns.

    Returns:
        True if the size and the parameters match, False otherwise.
    """
    if int(size) != int(param[0]) * int(param[1]):
        print("Error : Size and parameter do not match")
        return False
    return True

def one_param_test(size,param):
    """
    Test whether the given parameter is valid for the given size.
    Args:
        size: The size to test against.
        param: The parameter to test.
        
    Returns:
        True if the given parameter is valid for the given size, False otherwise.
    """
    if len(param) == 1:
        if int(param[0]) == int(size):
            return True
        else :
            print("Error : Size and parameter do not match")
            return False    
        print("Error : Incorrect number of parameters")
    return False

special_graph_tests = {
'GRID': (grid_test,2), 
'COMPLETE':(one_param_test,1), 
'LINE': (one_param_test,1), 
'CYCLE':(one_param_test,1) }

def modify_graph(type="COMPLETE", size=n0, param=[n0]):
    """
    Modify the graph configuration according to the specified type and parameters.
    
    Args:
        type: The type of graph to modify.
        size: The size of the modified graph.
        param: The parameters of the modified graph.
        
    Returns:
        True.
    """
    # Convert the type to uppercase
    type = type.upper()
    # Set the 'type' field in the 'GRAPHTYPE' section of the graph_config object
    graph_config.set('GRAPHTYPE', 'type', type)
    
    # Check that the number of parameters matches the number expected for the specified graph type
    if len(param) != len(graph_config[type+'PARAM'])-1:
        print("Error: number of parameters do not match")
        
    # Look up the test function and number of parameters for the specified graph type in the special_graph_tests dictionary
    test_func, num_params = special_graph_tests[type]
    # If the test function returns True, modify the graph configuration
    if test_func(size, param):
        # Set the 'num_nodes' field in the 'ALGOCONFIG' section of the param_config object
        param_config.set('ALGOCONFIG', 'num_nodes', str(size))
        # Set the fields in the type + 'PARAM' section of the graph_config object to the string representation of the elements in the param list
        for i in range(len(param)):
            graph_config.set(type+'PARAM', 'n'+str(i), str(param[i]))
    return True

def modify_round(T=T):
    """
    Modify the round parameter in the algorithm configuration.
    
    Args:
        T: The new value for the round parameter.
        
    Returns:
        True.
    """
    # Set the 't' field in the 'ALGOCONFIG' section of the param_config object to the string representation of T
    param_config.set('ALGOCONFIG', 't', str(T))
    return True

def modify_iterations(L=L):
    """
    Modify the number of iterations in the algorithm configuration.
    
    Args:
        L: The new value for the number of iterations.
        
    Returns:
        True.
    """
    # Set the 'l' field in the 'ALGOCONFIG' section of the param_config object to the string representation of L
    param_config.set('ALGOCONFIG', 'l', str(L))
    return True

def modify_num_trials(num_trials=num_trials):
    """
    Modify the number of trials in the experiment configuration.
    
    Args:
        num_trials: The new value for the number of trials.
        
    Returns:
        True.
    """
    # Set the 'num_trials' field in the 'EXPERIMENTCONFIG' section of the param_config object to the string representation of num_trials
    param_config.set('EXPERIMENTCONFIG', 'num_trials', str(num_trials))
    return True


def modify_batch_size(batch_size=batch_size):
    """
    Modify the batch size in the algorithm configuration.
    
    Args:
        batch_size: The new value for the batch size.
        
    Returns:
        True.
    """
    # Set the 'batch_size' field in the 'ALGOCONFIG' section of the param_config object to the string representation of batch_size
    param_config.set('ALGOCONFIG', 'batch_size', str(batch_size))
    return True


def modify_cifar10():
    """
    Modify the configuration parameters for the CIFAR-10 dataset.
    
    Returns:
        True.
    """
    # Set the 'dataset' field in the 'DATAINFO' section of the param_config object
    param_config.set('DATAINFO', 'dataset', 'sorted_cifar10.csv')
    # Set the 'f' field in the 'DATAINFO' section of the param_config object
    param_config.set('DATAINFO', 'f', '3072')
    # Set the 'c' field in the 'DATAINFO' section of the param_config object
    param_config.set('DATAINFO', 'c', '10')
    # Set the 'r' field in the 'ALGOCONFIG' section of the param_config object
    # Set the 'batch_size' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'batch_size', '500')
    # Set the 'l' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'l', '10')
    # Set the 't' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 't', '100')
    # Set the 'eta' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'eta', '0.1')
    # Set the 'eta_exp' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'eta_exp', '1')
    # Set the 'rho' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'rho', '1')
    # Set the 'rho_exp' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'rho_exp', '0.5')
    # Set the 'reg' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'reg', '100')
    # Set the 'eta' field in the 'FWCONFIG' section of the param_config object
    param_config.set('FWCONFIG', 'eta', '0.25')
    # Set the 'eta_exp' field in the 'FWCONFIG' section of the param_config object
    param_config.set('FWCONFIG', 'eta_exp', '1')
    # Set the 'l' field in the 'FWCONFIG' section of the param_config object
    param_config.set('FWCONFIG', 'l', '50')
    return True

def modify_mnist():
    """
    Modify the configuration parameters for the MNIST dataset.
    
    Returns:
        True.
    """
    # Set the 'dataset' field in the 'DATAINFO' section of the param_config object
    param_config.set('DATAINFO', 'dataset', 'sorted_mnist.csv')
    # Set the 'f' field in the 'DATAINFO' section of the param_config object
    param_config.set('DATAINFO', 'f', '784')
    # Set the 'c' field in the 'DATAINFO' section of the param_config object
    param_config.set('DATAINFO', 'c', '10')
    # Set the 'r' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'r', '8')
    # Set the 'batch_size' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'batch_size', '600')
    # Set the 'l' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'l', '10')
    # Set the 't' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 't', '100')
    # Set the 'eta' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'eta', '1')
    # Set the 'eta_exp' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'eta_exp', '1')
    # Set the 'rho' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'rho', '4')
    # Set the 'rho_exp' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'rho_exp', '0.5')
    # Set the 'reg' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'reg', '20')
    # Set the 'eta' field in the 'FWCONFIG' section of the param_config object
    param_config.set('FWCONFIG', 'eta', '1.5')
    # Set the 'eta_exp' field in the 'FWCONFIG' section of the param_config object
    param_config.set('FWCONFIG', 'eta_exp', '1.5')
    # Set the 'l' field in the 'FWCONFIG' section of the param_config object
    param_config.set('FWCONFIG', 'l', '50')
    return True

def modify_mfw():
    """
    Modify the configuration parameters for the MFW algorithm.
    
    Returns:
        True.
    """
    # Set the 'algo' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'algo', 'mfw')
    # Modify the graph configuration to be a complete graph with 1 node
    modify_graph(type="COMPLETE", size=1, param=[1])
    # Modify the number of trials to be 1
    modify_num_trials(num_trials=1)
    return True

def modify_rwmfw():
    """
    Modify the configuration parameters for the RWMFW algorithm.
    
    Returns:
        True.
    """
    # Set the 'algo' field in the 'ALGOCONFIG' section of the param_config object
    param_config.set('ALGOCONFIG', 'algo', 'rwmfw')
    # Modify the number of trials to be 50
    modify_num_trials(num_trials=50)
    return True


def sort_by_int(l):
    """
    Sort a list of strings containing integers in ascending order.
    
    Args:
        l: The list of strings to be sorted.
    
    Returns:
        A sorted list of strings.
    """
    # Convert the elements in the input list to integers
    tmp = [int(e) for e in l]
    # Sort the list of integers
    tmp = sorted(tmp)
    # Convert the sorted list of integers back to strings
    res = [str(e) for e in tmp]
    return res

def update_configs():
    """
    Update the configuration files with the modified configuration parameters.
    
    Returns:
        None.
    """
    # Write the modified graph_config object to the 'config/graph.conf' file
    with open('config/graph.conf', 'w') as configfile:
        graph_config.write(configfile)
    # Write the modified param_config object to the 'config/param.conf' file
    with open('config/param.conf', 'w') as configfile:
        param_config.write(configfile)

def exit_success(succ):
    """
    Print the success status and exit the program.
    
    Args:
        succ: The success status to be printed (True or False).
    
    Returns:
        None.
    """
    # Print the success status
    print(str(succ))
    # Exit the program
    exit()


def exit_error(msg):
    """
    Print an error message and exit the program.
    
    Args:
        msg: The error message to be printed.
    
    Returns:
        None.
    """
    # Print the error message
    print("Error: " + msg)
    # Exit the program
    exit()


if __name__ == "__main__":
    # Get the number of command-line arguments
    argc = len(sys.argv)
    # Initialize the modified flag to False
    modified = False
    
    # If there are more than 3 arguments
    if argc > 3:
        # If the first argument is 'G'
        if sys.argv[1].upper() == "G":
            # Check if all graph parameters are integers
            for i in range(3,argc):
                if not checkInt(str(sys.argv[i])):
                    exit_error("graph parameters should be integers")
            
            # If the graph type takes one parameter and there are exactly 4 arguments
            if special_graph_tests[sys.argv[2].upper()][1] == 1 and argc==4:
                # Modify the graph configuration
                modified = modify_graph(sys.argv[2].upper(), sys.argv[3], [sys.argv[3]])
            
            # If there are more than 4 arguments
            if argc > 4:
                # Modify the graph configuration
                modified = modify_graph(sys.argv[2].upper(), sys.argv[3], sort_by_int(sys.argv[4:]))
            
            # If the modification was successful
            if modified:
                # Update the configuration files
                update_configs()
                # Exit the program with success status
                exit_success(1)
    
    # If there are exactly 3 arguments
    if argc == 3:
        # If the first argument is 'T'
        if sys.argv[1].upper() == "T":
            # Check if the number of rounds is an integer
            if not checkInt(str(sys.argv[2])):
                exit_error("number of rounds T should be integer")
            # Modify the number of rounds
            modified = modify_round(sys.argv[2])
        # If the first argument is 'L'
        if sys.argv[1].upper() == "L":
            # Check if the number of iterations is an integer
            if not checkInt(str(sys.argv[2])):
                exit_error("number of iterations L should be integer")
            # Modify the number of iterations
            modified = modify_iterations(sys.argv[2])
        # If the first argument is 'TRIALS'
        if sys.argv[1].upper() == "TRIALS":
            # Check if the number of trials is an integer
            if not checkInt(str(sys.argv[2])):
                exit_error("number of iterations L should be integer")
            # Modify the number of trials
            modified = modify_num_trials(sys.argv[2])
        # If the first argument is 'BS' or 'BATCH_SIZE'
        if sys.argv[1].upper() == "BS" or sys.argv[1].upper() =="BATCH_SIZE":
            # Check if the batch size is an integer
            if not checkInt(str(sys.argv[2])):
                exit_error("the size of batch should be integer")            
        # Modify the batch size
        modified = modify_batch_size(sys.argv[2])
        # If the first argument is 'ALGO'
        if sys.argv[1].upper() == "ALGO":
            # If the second argument is 'MFW'
            if sys.argv[2].upper() == "MFW":
                # Modify the parameters for the MFW algorithm
                modified = modify_mfw()
            # If the second argument is 'RWMFW'
            if sys.argv[2].upper() == "RWMFW":
                # Modify the parameters for the RWMFW algorithm
                modified = modify_rwmfw()
            # If the second argument is 'DMFW'
            if sys.argv[2].upper() == "DMFW":
                # Modify the parameters for the DMFW algorithm
                modified = modify_dmfw()
        # If any modification was made
        if modified :
            # Update the configuration files
            update_configs()
            # Exit the program with success
            exit_success(1)
        # If no modification was made, exit the program with an error message
        exit_error("Incorrect argument.")
    # If there is only one argument
    if argc == 2:
        # If the argument is 'MNIST'
        if sys.argv[1].upper() == "MNIST":
            # Modify the parameters for the MNIST dataset
            modified = modify_mnist()
        # If the argument is 'CIFAR10'
        if sys.argv[1].upper() == "CIFAR10":
            # Modify the parameters for the CIFAR10 dataset
               modified = modify_cifar10()
        # If any modification was made
        if modified :
            # Update the configuration files
            update_configs()
            # Exit the program with success
            exit_success(1)
        # If no modification was made, exit the program with an error message
        exit_error("Incorrect argument")
    # If no modification was made, exit the program with an error message
    exit_error("Not enough arguments !")
