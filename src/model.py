import numpy as np
import json
from brian2 import *
from brian2.units.constants import faraday_constant as F

def set_up_scope(delta_t:float = 0.01, duration: float=1/3):
    """
    Sets up the simulation environment by initializing a new simulation scope
    with the given time step and duration.

    Args:
        delta_t (float, optional): The time step of the simulation,
        in seconds. Defaults to 0.01.
        duration (float, optional): The duration of the simulation,
        in minutes. Defaults to 1/3.

    Returns:
        Quantity: The duration of the simulation as a Quantity object
        with units of seconds.
    """
    start_scope()
    defaultclock.dt = delta_t * ms
    # Duration in minutes
    total = int(duration * 60000)
    duration = total * ms
    return duration


def get_model_equations():
    """
    Returns the model equations.
    Add reference paper
    """
    eqs = """
        DNa_i = -DK_i : 1 (constant over dt)
        DNa_o = -beta * DNa_i : 1 (constant over dt)
        DK_o = -beta * DK_i : 1 (constant over dt)
        K_i = K_i0 + DK_i : 1 (constant over dt)
        Na_i = Na_i0 + DNa_i : 1 (constant over dt)
        Na_o = Na_o0 + DNa_o : 1 (constant over dt)
        K_o = K_o0 + DK_o + Kg : 1 (constant over dt)

        n_inf= 1.0/(1.0+exp((Cnk-V)/DCnk)): 1
        m_inf= 1.0/(1.0+exp((Cmna-V)/DCmna)): 1
        h_n= 1.1 - 1.0 / (1.0 + exp(DChn * (n - Chn))): 1

        I_K   = (g_Kl+g_K*n)*(V- 26.64*log(K_o/K_i)) : 1 
        I_Na  = (g_Nal+g_Na*m_inf*h_n)*(V- 26.64*log(Na_o/Na_i)) : 1 
        I_Cl  = g_Cl*(V+ 26.64*log(Cl_o0/Cl_i0)) : 1 
        I_pump= rho*(1.0/(1.0+exp((Cnap - Na_i) / DCnap))*(1.0/(1.0+exp((Ckp - K_o)/DCkp)))) : 1 

        dV/dt = (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump)+(eta)/tau_I: 1
        dn/dt = (n_inf - n) / tau_n : 1
        dDK_i/dt = -(gamma/w_i) * (I_K-2*I_pump): 1
        dKg/dt = epsilon * (K_bath - K_o) : 1

        Cnap : 1
        DCnap : 1
        Ckp : 1
        DCkp : 1
        Cmna : 1
        DCmna : 1
        Chn : 1
        DChn : 1
        Cnk : 1
        DCnk : 1
        Cm : second
        tau_n : second
        tau_I : second
        g_Cl : 1
        g_Na : 1
        g_K : 1
        g_Nal : 1
        g_Kl : 1
        w_i : 1
        w_o : 1
        rho : 1
        beta : 1
        epsilon : hertz
        gamma : hertz
        K_bath : 1
        Na_i0 : 1
        Na_o0 : 1
        K_i0 : 1
        K_o0 : 1
        Cl_o0 : 1
        Cl_i0 : 1
        E : 1
        J: 1
        eta:1
    """
    return eqs


def create_cauchy_samples(N, Delta, eta_bar):
    """
    Generates a Cauchy distribution with location parameter eta_bar and
    scale parameter Delta,  discarding any samples outside the range
    (-20 + eta_bar, 20 + eta_bar), and adding eta_bar 
    to fill in any missing values.

    Args:
        N (int): The number of samples to generate.
        Delta (float): The scale parameter of the Cauchy distribution.
        eta_bar (float): The location parameter of the Cauchy distribution.

    Returns:
        np.ndarray: An array of N Cauchy-distributed samples with location
        parameter eta_bar and scale parameter Delta.
    """
    s = np.random.standard_cauchy(N) * Delta + eta_bar
    s = s[(s > -20 + eta_bar) & (s < 20 + eta_bar)]
    s = np.append(s, eta_bar * np.ones(N - len(s)))
    return s

def read_parameters(path="parameters.json"):
    """
    Read json file with default parameter values.
    """
    with open(path) as f:
        parameters = json.load(f)
    return parameters

def create_neuron_group(N: int = 100, threshold: str="V > Vth",
                        refractory: str="V>Vth", method="heun",
                        init_states:list=None, k_bath: float = 15.5,
                        eta_bar: float = 0.1, Delta: float = 1.0,
                        connection_rule:str="i!=j",
                        parameters_path:str="parameters.json",
                        verbose:bool=False):
    """
    Creates a neuron group using Brian2, with customizable parameters.

    Parameters
    ----------
    N : int, optional
        The number of neurons in the group. Default is 100.
    threshold : str, optional
        The threshold condition for the neuron group. Default is "V > Vth".
    refractory : str, optional
        The refractory condition for the neuron group. Default is "V > Vth".
    method : str, optional
        The integration method for the neuron group. Default is "heun".
    init_states : list, optional
        The initial states of each of the four initial variables. 
        Must be a list with four elements. Default is None.
    k_bath : float, optional
        The bath concentration of potassium ions in mM. Default is 15.5.
    eta_bar : float, optional
        The mean value of the Cauchy distribution used to generate the noise
        in the model. Default is 0.1.
    Delta : float, optional
        The width of the Cauchy distribution used to generate the noise in
        the model. Default is 1.0.
    connection_rule : str, optional
        The connection rule used to connect the neurons in the group.
        Default is "i!=j".
    parameters_path : str, optional
        The path to the file containing default parameter values for the model.
        Default is "parameters.json".
    verbose : bool, optional
        If True, print additional information about the model. Default is False.

    Returns
    -------
    tuple
        A tuple containing the neuron group, the Brian2 network,
        the state monitor, the spike monitor, and the
        population rate monitor.

    """
    # The initial state of each of four initial variables
    assert len(init_states) == 4
    # Assure it is a numpy array
    init_states = np.asarray(init_states)

    # Get model set of equations
    eqs = get_model_equations()
    # Define neuron group
    group = NeuronGroup(N, eqs, threshold=threshold,
                        refractory=refractory,
                        method=method)

    state_variables = ["V", "n", "DK_i", "Kg"]
    # Add variability to initial state
    init_states = np.random.normal(size=(N, 4)) * 0.1 + init_states
    for p, state_var in enumerate(state_variables):
        group.set_states({state_var: init_states[:, p]}, units=False)

    # Read file with default parameter values
    parameters = read_parameters(parameters_path)
    # Initialize parameters of the model
    for key in parameters.keys():
        unit = 1
        if key in ["epsilon", "gamma"]:
            unit = (1 / ms)
        if key in ["tau_I", "tau_n", "Cm"]:
            unit = ms
        has_units = isinstance(unit, units.fundamentalunits.Unit)
        group.set_states({key: parameters[key] * unit}, units=has_units)

    group.J = 1.0 / N
    group.beta = group.w_i / group.w_o
    # Init eta
    group.eta = create_cauchy_samples(N, Delta, eta_bar)
    # Init kbath
    group.K_bath = k_bath

    # Create monitors
    # state_monitor = StateMonitor(group, ("V", "n", "DK_i", "Kg", "K_o"),
                                 # record=True, dt=1 * ms)
    # spike_monitor = SpikeMonitor(group)
    # fr_monitor = PopulationRateMonitor(group)

    # Create synapses
    syn = Synapses(group, group, on_pre="V+=J*(E-V) ", method=method)
    syn.connect(connection_rule)

    # Create network
    net = Network(collect())
    # net.add(group, syn, state_monitor, spike_monitor, fr_monitor)

    if verbose:
        print("K_bath=%.2f" % group.K_bath[0])
        print("J=%.3f" % group.J[0])

    start_scope()
    run(1000 * ms, report="stdout", report_period=30 * second)

    return group, net#, state_monitor, spike_monitor, fr_monitor
