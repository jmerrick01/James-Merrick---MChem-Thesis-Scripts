# Coulomb Explosion Simulation Program 2021
# - simulates positions of resultant ions from coulomb explosion
# Prepare an input file with a description of the fragmentation pattern then run this program.
# for more info contact Louis Minion (louisalvar@gmail.com)

# Modules
import numpy as np
import re
import os
import time
import numpy as np
import argparse
import re
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.spatial.transform import Rotation
import itertools
import numpy.random
from numpy.random import default_rng
import scipy.stats
from scipy.stats import truncnorm
s_time = time.time()

## Functions

def symbol_to_weight(symbol):
    """
    Returns atomic mass of element with given symbol.
    Wrapper for mendeleev get_all_elements.

    Parameters
    -----------
    symbol: element symbol

    Returns
    -----------
    w: atomic mass
    """
    symbol = str(symbol)
    import mendeleev as m
    els = m.get_table('elements')
    cols = ['atomic_weight', 'symbol']
    ptable = els[cols]
    w = ptable.loc[ptable.symbol == symbol].atomic_weight
    w = float(w)
    return w
def atom_number_to_mol_weight(atom_number):
    """
        Returns atomic mass of element with given atomic number.
        Wrapper for mendeleev get_all_elements.

        Parameters
        -----------
        atom_number: atomic number

        Returns
        -----------
        w: atomic mass
    """
    import mendeleev as m
    els = m.get_all_elements()
    weight = els[atom_number-1].atomic_weight
    return weight
## Extracts geometry of a molecule (cartesians) in Angstroms plus masses in Daltons
def get_geom(filename):
    """
        Scans through a Gaussian log file for an optimised geometry
        and puts in dataframe. Looks for the phrase
        'Standard Orientation' in a log file, which preceeds the output
        cartesians of an optimised structure.
        The output df has a row for each atom of the form;
        mass x y z
        Works for Gaussian 09 and Gaussian 16 log files. Suggest update
        so instead looks through Gaussian fchk file for forwards compatability.

        Parameters
        -----------
        filename: log file

        Returns
        -----------
        df: pandas DataFrame with m,x,y,z info for each atom.

        Raises
        ----------
        Exceptions if;
        - Regex does not find route section, can't check if it is an optimisation job.
        - Route section does not contain the phrase 'opt' - ie only takes optimisation jobs.
    """
    import numpy as np
    import pandas as pd
    # print(filename)
    with open(filename, 'r') as file:
        whole_file = file.read()
    try:
        found = re.search('#(.+?)\n', whole_file).group(1)
    except AttributeError:
        raise Exception('No route section found. Is this a Gaussian file?')
    # print('Gaussian route: {}'.format(found))
    if found.find('opt') != -1:
        pass
    else:
        raise Exception('Geometry extraction requires an optimisation job as input')
    start = 0
    end = 0
    openold = open(filename,"r")
    rline = openold.readlines()
    for i in range (len(rline)):
        if "Standard orientation:" in rline[i]:
            start = i
        # Data starts 5 lines below this statement and ends at ---
    for m in range (start + 5, len(rline)):
        if "---" in rline[m]:
            end = m
            break
    df = pd.DataFrame(columns=['a', 'x', 'y', 'z'])
        # words list 0 - number, 1 - atomic number, 2- atom type, 3, 4, 5 - x,y,z
    for line in rline[start+5 : end] :
        words = line.split()
        word1 = int(words[1])
        word1 = atom_number_to_mol_weight(word1)
        words[1] = word1
        wdf = pd.DataFrame([[words[1], words[3], words[4], words[5]]], columns=['a', 'x', 'y', 'z'])
        df = df.append(wdf)
    df = df.astype(np.float64)
    return df

### Takes a gaussian freq job file and returns dictionaries containing normal modes and frequencies
def extract_normal_modes(filename, N_atoms):
    """
            Scans through a Gaussian log file for frequency information.
            Normal modes given after the phrase 'and normal coordinates', which
            regex searches for.
            Unfortunately it appears that Gaussian fchk does not at the moment contain
            normal modes in cartesians, so this workaround is used here. May become obsolete
            if Gaussian changes the format of log files.

            Parameters
            -----------
            filename: log file (must be from calculation ran with P option after # in route.)

            Returns
            -----------
            modes: dict with numbers as keys and a list of lists as values. These lists correspond to the normal
                   mode vectors (as a list of x,y,z elements) for each atom - ie the directions in Cartesian space
                   each atom needs to move to correspond to the vibration along the mode given by the key.

            freqsdict: dict with numbers as keys and frequencies (in wavenumbers) as values.
                       Keys correspond to those in the modes dict, so freqsdict[key] gives the
                       frequency of vibration along mode modes[key].
            massesdict: dict of the reduced masses corresponding to the modes found.

            Raises
            ----------
            Exceptions if;
            - Regex does not find route section, can't check if it is an optimisation job.
            - Route section does not contain the phrase 'freq' - ie only takes optimisation jobs.
            - Route section does not contain 'P' - log files only contain normal mode cartesians if this option is used in the calculation.
    """
    import numpy as np
    import re
    file_type = filename[-3:]
    if file_type != 'log':
        raise Exception('Must be Gaussian log file')
    calc_name = filename.split('\\')[-1][:-4]

    with open(filename, 'r') as file:
        whole_file = file.read()
# Finding route section from output file
    try:
        found = re.search('#(.+?)\n', whole_file).group(1)
    except AttributeError:
    # cmd not found in the original string
        found = 'No route section'
        raise Exception('No route section found whilst extracting vibrational information')
    if found.find('Freq') != -1:
        pass
    else:
        raise Exception('Not a frequency calculation from Gaussian')
    # if found[0] != 'P':
    #     raise Exception('Must run Gaussian with P option in route for extra output in log file')

    # print('Searching for Normal Modes')
    start = 'and normal coordinates:\n'
    end = ' -------------------'
    normal_string = whole_file.split(start)[1].split(end)[0]
    rows = normal_string.split('\n')
    N_vibmodes = 3*(len(rows) - 2)/(7+N_atoms)
    blocks = N_vibmodes/3
    blocklength = 7+N_atoms
    rows = rows[:-2]
    rws = [rows[i:i + blocklength] for i in range(0, len(rows), blocklength)]
    modes = {}
    freqsdict = {}
    massesdict = {}
    for block in rws:
        moderefs = block[0].split()
        for num in moderefs:
            modes[num] = [ [] for d in range(N_atoms) ]
        for num in moderefs:
            freqsdict[num] = None
        modesymms = block[1].split()
        modefreqs = block[2].split()
        modefreqs = list(map(float, modefreqs[-3:]))
        print(modefreqs[-3:])
        for g in range(0, len(modefreqs[-3:])):
            freqsdict[moderefs[g]] = (modefreqs[-3:])[g]
        moderedmasses = block[3].split()
        modeforceconstants = block[4].split()
        modeIRintensity = block[5].split()
        moderedmasses = list(map(float, moderedmasses[-3:]))
        for g in range(0, len(moderedmasses[-3:])):
            massesdict[moderefs[g]] = (moderedmasses[-3:])[g]
        normlmodes = block[7:]

        for i in range(0, len(normlmodes)):
            atom = normlmodes[i].split()
            atomno = int(atom[0])
            coords = atom[2:]
            coords = [coords[i:i + 3] for i in range(0, len(coords), 3)]
            for j in range(0, 3):
                vec = coords[j]
                vec = list(map(float, vec))
                modes[moderefs[j]][i] = vec



    return modes, freqsdict, massesdict

class Experiment:
    """
    class Experiment
    Wrapper object that holds all the needed
    information to run a simulation. Created by
    'read_instructions'.

    attributes
    -----------
     - Experiment.compchemfile: Gaussian filename specifying geometry and
       vibrational information.
     - Experiment.geometry: pandas Dataframe returned by get_geom.
     - Experiment.molecule_object: instance of the molecule class with
       the geometry required.
     - Experiment.numatoms: number of atoms in the molecule being simulated.
     - Experiment.modes, Experiment.freqsdict: see extract_normal_modes

     methods
     -----------
     - Experiment.assigncharges(self, channels)
        Used for handling the instructions file and grouping atoms into fragments
        for different fragmentation pathways.
        Parameters
        -----------
        - channels: list of specifications for each
          fragmentation channel passed by read_instructions.
        -----------
        Creates:
        - Experiment.channel_ions: dict of dicts. Parent
          dict keys are channel numbers. Subdict keys are
          fragment numbers. Subdict values are lists of atom
          objects which should be grouped into that fragment.
          Allows specification of many different fragmentation
          patterns in one simulation.
        - Experiment.channel_probs: dict, keys are channel numbers,
          values are the probabilities of that channel compared to
          all others. These must sum to 1.
        - Experiment.channel_charges: dict of dicts. Parent dict keys are
          channel numbers, subdict keys are fragment numbers. Subdict values
          are charges each fragment should have in units of e.
    - Experiment.createMoleculeObj(self)
        Returns a copy of the attribute
        self.molecule_object to retain equilibrium geometry
        in the parent Experiment object, and allow the copied
        molecule object to be rotated, vibrated etc.
        Returns
        -----------
        mol: molecule instance with geometry given by Experiment.geometry
    - Experiment.normaldist(self, sigma, upper, lower, mu)
        Initialises truncated normal distribution for augmented Poisson distribution.
        Upper and lower are bounds of the distribution, upper set very high (1000) and lower
        set to 0 to prevent unphysical negative event rate.
        sigma is the experimental fluctuation parameter, mu is always 1.
        Returns
        -----------
        Gamma_dist: scipy.stats frozen rv_continuous instance.
    """

    def __init__(self, filename):
        self.compchemfile = filename
        self.geometry = get_geom(filename)
        self.molecule_object = molecule(self.geometry)
        self.molecule_object.group_atoms()
        self.numatoms = len(self.molecule_object.atoms)
        self.modes, self.freqsdict, self.redmassesdict = extract_normal_modes(self.compchemfile, self.numatoms)
    def __repr__(self):
        reprstring = 'Experiment("{}") Object'.format(self.compchemfile)
        return reprstring
    def assigncharges(self, channels):
        channel_ions = {i : None for i in range(0, len(channels))}
        channel_probs = {i : None for i in range(0, len(channels))}
        channel_charges = {i : None for i in range(0, len(channels))}
        N = self.numatoms
        for i in range(0, self.no_channels):   # loop over all channels channels is a list of lists, each one the specification for an individualchannel
            no_fragments = channels[i][0]   # set var no_fragments
            no_fragments = int(no_fragments)
            prob = channels[i][1]
            channel_probs[i] = prob
            frag_assignment = channels[i][2:-no_fragments] # the part of the list with which fragment atoms belong to
            charges = channels[i][-no_fragments:] # the part of the list specifying charges
            atoms_in_fragment = {i : None for i in range(1, no_fragments+1)} # empty dict with key for each fragment
            atoms = self.molecule_object.atoms
            for j in range(0, N):
                frag_assignment[j] = int(frag_assignment[j])
                atoms[j].frag_no = frag_assignment[j] # assign a fragnumber to each atom
            fragment_atoms = {k: [] for k in range(1, no_fragments+1)}
            for k in range(1, no_fragments+1): # over fragments
                for l in range(0, N): # over atoms
                    if atoms[l].frag_no == k: #check fragnumber
                        fragment_atoms[k].append(l)
                    else:
                        pass
            channel_ions[i] = fragment_atoms
            fragment_charges = {k: None for k in range(1, no_fragments+1)}
            for m in range(0, no_fragments):
                charges[m] = int(charges[m])
                fragment_charges[m+1] = charges[m]
            channel_charges[i] = fragment_charges
        channel_probs = list(channel_probs.values())
        channel_probs = list(np.float_(channel_probs))
        self.channel_ions = channel_ions
        self.channel_probs = channel_probs
        self.channel_charges = channel_charges

    def createMoleculeObj(self):
        from copy import deepcopy
        # mol = molecule(self.geomfile)
        mol = deepcopy(self.molecule_object)
        mol.group_atoms()
        # inherit vib motion info
        mol.freqsdict = deepcopy(self.freqsdict)
        mol.modes = deepcopy(self.modes)
        mol.redmasses = deepcopy(self.redmassesdict)
        return mol
    def normaldist(self, sigma, upper, lower, mu):
        if sigma == 0:
            # print('No fluctuation in event rate case')
            pass
        else:
            Gamma_dist = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            self.Gamma_dist = Gamma_dist
            return Gamma_dist
class atom:
    """
    class atom
    Used to keep x,y,z, mass information for an atom in the same place.
    """
    def __init__(self, x, y, z, m):
        self.mass = m
        self.pos = np.array([x,y,z])

class molecule:
    """
    class molecule(geometry)
    - geometry: a DataFrame with m,x,y,z info.
    Instances created dynamically by Experiment object.
    Holds geometry of all atoms, can have their positions
    vibrate, can 'explode' by creating charged fragment objects.

    attributes
    -----------
    - molecule.info: df given by geometry
    - molecule.masses: array of masses for each atom
    - molecule.x_coords, y_coords, z_coords: ditto for Cartesians.

    methods
    -----------
    - group_atoms(self)
      Creates atom instances for every atom in the molecule.
      Stores them as a list called self.atoms
      Preserves equilibrium positions by storing as self.equilibrium_pos
    - vib(self)
      Samples probability distribution and moves along each of the normal modes
      the atomic coordinates by a given amount. Distribution centred at eq pos.
      Updates the pos attributes of molecule.atoms to correspond to the vibrated
      position.
    - createFragments(self, channel, expmnt)
      Groups the current atomic positions into fragment objects and assigns a charge to each.
      Parameters
      -----------
      - channel: which fragmentation pathway to create fragments for.
      - expmnt: parent Experiment instance containing required information.
      Creates
      ----------
      molecule.ions: a list of fragment objects assigned charges,
       at the positions inherited from molecule.atoms

    """
    import pandas as pd
    def __init__(self, geometry):
        self.info = geometry
        self.masses = np.array(self.info)[:, 0]
        self.x_coords = np.array(self.info)[:, 1]
        self.y_coords = np.array(self.info)[:, 2]
        self.z_coords = np.array(self.info)[:, 3]
    def __repr__(self):
        rep = 'molecule object'
        return rep
    def group_atoms(self): # creates atom objects which describe the positions and masses of each atom in the molecule
        atoms = []
        for i in range(0, len(self.x_coords)):
            a = atom(self.x_coords[i], self.y_coords[i], self.z_coords[i], self.masses[i])
            atoms.append(a)
        self.atoms = atoms
        eq_pos = []
        for atm in self.atoms:
            eq_pos.append(atm.pos)
        self.equilibrium_pos = eq_pos
    def vib(self):
        from scipy.constants import hbar, u, c
        from scipy.stats import norm
        atoms = self.atoms
        eq_pos = self.equilibrium_pos
        eq_pos = np.array(eq_pos)
        for k in list(self.modes.keys()):
            directions = self.modes[k]
            freq = self.freqsdict[k]
            freq = freq*c*100
            redmass = self.redmasses[k]
            redmass = redmass*u
            classical_turning_point = np.sqrt(hbar/(freq*redmass)) # hbar/w*mu
            classical_turning_point = classical_turning_point*(10**10) # metres to angstroms
            wavefunction_absolutesquare_sigma = classical_turning_point/2
            probd = norm(loc=0, scale=wavefunction_absolutesquare_sigma)
            sample = probd.rvs()
            for i in range(0, len(atoms)):
                direction = np.array(directions[i])
                eq_pos[i] = eq_pos[i] + sample*direction
        for i in range(0, len(atoms)):
            atoms[i].pos = eq_pos[i]
        self.atoms = atoms
    def createFragments(self, channel, expmnt):
        channel_ions = expmnt.channel_ions
        channel_charges = expmnt.channel_charges
        atoms_in_fragment = channel_ions[channel]
        charges = channel_charges[channel]
        atoms = self.atoms
        no_fragments = max(atoms_in_fragment, key=int)
        fragmentdicts = {i : None for i in range(1, no_fragments+1)}
        ions = []
        for j in range(1, no_fragments+1):
            atomlist = atoms_in_fragment[j]
            tempatoms = []
            for i in range(0, len(atoms)):
                if i in atomlist:
                    tempatoms.append(atoms[i])
                else:
                    pass
            frag = fragment(tempatoms)
            frag.centre_of_mass()
            frag.q = charges[j]
            ions.append(frag)
        self.ions = ions

class fragment:
    """
    class fragment
    - atoms_list: list of atom objects which make up the fragment.
    These are the objects which are passed to the ODE solver - the ionic
    fragments produced from a parent molecule in a Coulomb explosion.
    Stores total mass, charge, and a centre of mass position.
    """
    def __init__(self, atoms_list):
        self.atoms = atoms_list
        def total_mass(atoms_list):
            N = len(atoms_list)
            mass = 0
            for i in range(0, N):
                mass = mass + atoms_list[i].mass
            return mass
        self.mass = total_mass(atoms_list)
    def centre_of_mass(self):
        atoms_list = self.atoms
        N = len(atoms_list)
        if N == 1:
            self.pos =  atoms_list[0].pos
        else:
            mass = 0
            sum = 0
            for i in range(0, N):
                mass = mass + atoms_list[i].mass
                sum = sum + (atoms_list[i].mass * atoms_list[i].pos)
            c_o_m = (1/mass)*sum
            self.pos = c_o_m

def read_instructions(file):
    """
    Reads a text file provided by the user which contains information
    on simulation parameters. Relies on a specific syntax in this instruction file.
    Parameters
    ----------
    file: text filename.

    Returns
    ---------
    Experiment object containing the information required for the simulation.

    Raises
    --------
    - Exception -no END at the end of the instructions file.
    """
    import pandas as pd
    import numpy as np
    with open(file) as f:
        instructions = []
        for line in f:
            line = line.split('#', 1)[0]  # allow comments
            line = line.rstrip()
            instructions.append(line)
    name = instructions[0]
    compchemfile = instructions[1]
    species = Experiment(compchemfile)
    no_channels = int(instructions[2])
    species.no_channels = no_channels
    species.name = name
    v0 = int(instructions[3])
    species.v0 = v0
    ### Check
    if instructions[-1] == 'END':
        instructions = instructions[:-1]
    else:
        raise Exception('Enter END at the end of your input file')

    channel_split = splitlistbydelimiter(instructions, 'OUTCOME')
    channels = []
    for channel in channel_split:
        channels.append(channel)
    channels = channels[1:len(channels)]  # exclude 0th element as not a channel
    species.assigncharges(channels)
    print('Filename: '+str(compchemfile)+', Channels: '+str(no_channels)+', Event Rate: '+str(v0)+', Modes: '+str(len(species.modes)))
    return species





##################################################
# FUNCTIONS
def get_listofarrays(l, n=3):
    """
    splits a list l into a list containing arrays of length 3
    Used to split up output from ODE solver into 3-element vectors
    """
    list_of_arrays = [l[i:i+n] for i in range(0, len(l), n)]
    return list_of_arrays

def list_of_lists_to_list_of_series(ls):  # useful space saver
    """
    Wrapper for easily transforming output of simulation to df
    """
    s = []
    for l in ls:
        s0 = pd.Series(l)
        s.append(s0)
    return s

def splitlistbydelimiter(lst, seperator):    # generator object to split list by a delimiter, 'instructions' must be a list
    """
    generator object - splits a list containing separator elements (delimiters) into sublists
    Used to split the instructions input by the key-word OUTCOME
    """
    sublist = []
    for line in lst:
        if line == seperator:
            yield sublist
            sublist = []
        else:
            sublist.append(line)
    yield sublist

def forcecalc_ode(current_state, particles):
    """
    Takes the current state of the system, updates ion positions and recalculates forces at that new position.
    Calculates the total forces experienced by each "charged particle" object and returns them.
    Called by equations() func, which itself is called by the ODE solver solve_ivp.
    Parameters
    ---------
    - current_state: array of length 2*3N where N is the number of ions. Contains positions and velocities of
      each ion at the current step in the integration routine in the form [x1,y1,z1,...,xN,yN,zN, vx1,vy1,vz1,...,vxN,vyN, vzN]
    - particles: list of objects with attributes self.q (charge), self.mass (mass) and self.pos.
      Used to keep track of which ion has which charge and mass, as these are not automatically passed
      by solve_ivp. Charges passed to this function must be in units of elementary charges.

    Returns
    ---------
    - particles: list of the objects representing ions with updated forces and positions.
    """
    from scipy.constants import epsilon_0, e
    k = e**2/(4*np.pi*epsilon_0)
    pos, vel = np.split(current_state, 2)
    N = len(particles)
    for i in range(0, N):
        particles[i].force = np.zeros(3)

    pos_vectors = get_listofarrays(pos)
    for i in range(0,N):
        r = pos_vectors[i]  #sets r = (x_i, y_i, z_i)
        particles[i].pos = r
        if len(particles[i].pos)==3:
            pass
        else:
            raise Exception('Position vectors too long/short')   #sanity check
    for i in range(0, N):
        ion_i = particles[i]
        m_i = ion_i.mass
        r_i = ion_i.pos

        q_i = ion_i.q
        for j in range(i + 1, N):
            ion_j = particles[j]
            m_j = ion_j.mass
            q_j = ion_j.q
            r_j = ion_j.pos
            rvec = r_i - r_j
            rnorm = np.linalg.norm(rvec)
            F = ((k*q_i*q_j) / (rnorm**3)) * rvec
            ion_i.force = np.add(ion_i.force, F)
            ion_j.force = np.subtract(ion_j.force, F)
    return particles


def equations(t, current_state, *args):
    """
    Function called by the ODE solver solve_ivp
    Takes a current state of the system, (updated positions and velocities)
    and returns the rhs of the differential equations; dr/dt = v, return velocities for the positions,
    dv/dt = a, return accelerations for the velocities.
    Calls forcecalc_ode to get forces at the updated positions, then calculates accelerations
    using F = ma. solve_ivp does the actual integration, this just returns the values of the rhs of 1st order ODEs.
    Parameters
    ----------
    - t: required to be passed by solve_ivp, not actually used as none of the ODEs have functions of time.
    - current_state: positions and velocities whose derivatives are to be evaluated. Passed as array in the form
      [x1,y1,z1,...,xN,yN,zN, vx1,vy1,vz1,...,vxN,vyN, vzN].
    - args: these should be the 'fragment' objects whose masses and charges are needed to calculate forces and accelerations.
            Masses must be in gmol-1, charges in units of e.

    Returns
    --------
    - derivatives: array of form [dx1/dt, dy1/dt, dz1/dt...dxN/dt,dyN/dt,dzN/dt, dvx1/dt, dvy1/dt, dvz1/dt,...,dvxN/dt, dvyN/dt, dvzN/dt]
      evaluated at the positions and velocities given by current_state.

    """
    particles = []
    for i in args:        # remaking particles list
        particles.append(i)
    N = len(particles)
    particles = forcecalc_ode(current_state, particles) #calc forces at these positions and pass back to particle objects
    accelerations = []
    from scipy.constants import physical_constants
    u = physical_constants['atomic mass unit-kilogram relationship'][0]
    for i in range(0, N):
        a = particles[(i)].force/(particles[(i)].mass*u)
        accelerations.append(a[0])
        accelerations.append(a[1])
        accelerations.append(a[2])
    accelerations = np.array(accelerations)
    pos, vel = np.split(current_state, 2)
    Npos = len(pos)
    Npos2 = len(vel)
    if Npos == Npos2:
        pass
    else:
        raise Exception('vel and pos vectors not same length')
    derivatives = []
    for i in range(0, Npos):
        derivatives.append(vel[i])
    for i in range(0, Npos):
        derivatives.append(accelerations[i])
    derivatives = np.array(derivatives)
    return derivatives

def solve_equations(particles, t_to_return=np.array([0.0000000001]), t_span = (0.0,np.array([0.0000000001]))):
    """
    Wrapper function for solve_ivp. Used to take input in the form of a list of fragment objects
    (particles) and put it into the required form for the solver. t_to_return default value of a nanosecond lower limit
    on time for a particle to reach a detector if travelling at thousands of ms-1. Coulomb force scales as
    1/r^2 so most forces are very small by this point with light ions.
    Parameters
    ----------
    - particles: list of objects with pos, vel, mass and q (charge) attributes. (fragment instances)

    Returns
    -----------
    - sol: scipy.integrate Bunch object, which contains the info about the systems final state.

    Raises
    ----------
    - Exception: will print the message for why the solver failed.
    """
    N = len(particles)
    initial_values = []

    for i in range(0,N):
        initial_values.append(particles[i].pos[0])
        initial_values.append(particles[i].pos[1])
        initial_values.append(particles[i].pos[2])
    for i in range(0,N):
        initial_values.append(particles[i].vx)
        initial_values.append(particles[i].vy)
        initial_values.append(particles[i].vz)
    initial_values = np.array(initial_values)
    sol = solve_ivp(equations, t_span, initial_values, t_eval=t_to_return, args=particles)
    if sol.success == True:
        pass
    else:
        raise Exception('Solver failed, {}'.format(sol.message))
    return sol

def calc_time_of_flight(q, mass, flight_tube_length=0.5, voltage=1000):
    """
    Used to separate ions in time. Default values not accurate to
    a VMI setup but serve the correct purpose.
    """
    e = 1.60217662*10**-19
    u = 1.66053906660*10**-27
    t = flight_tube_length*np.sqrt((mass*u)/(2*q*e*voltage))
    return(t)

def rotate_molecule(particles, randstate):
    """
    For a list of position vectors defined by particles[i].pos, applies
    the same random rotation vector to all of them, to reflect a molecule
    rotating in space.
    Parameters
    ---------
    - particles: list of objects with an attribute pos
    - randstate: seed for random number generator. Needs new seed every call as
      creating a new Rotation object every time, and rng will start from the same place each time.
    Returns
    --------
    - particles: the same list of objects, with rotated pos vectors. All rotated the same amount in the
      same directions.
    """
    from scipy.spatial.transform import Rotation
    rotator = Rotation.random(random_state=randstate)
    N = len(particles)
    for i in range(0, N):
        particles[i].pos = rotator.apply(particles[i].pos)
    return particles
def angstroms_to_metres(particles):
    """
    returns particles with positions changed to metres (input in angstroms).
    """
    N = len(particles)
    particles = np.array(particles)
    for i in range(0, N):
        particles[i].pos = particles[i].pos*(10**-10)
    return particles

def set_vel_elements(particles):
    """
    Take list of ions and update vx, vy, vz attributes to represent vel attribute.
    """
    N = len(particles)
    for i in range(0,N):
        particles[i].vx = particles[i].vel[0]
        particles[i].vy = particles[i].vel[1]
        particles[i].vz = particles[i].vel[2]
    return particles

def init_vel(particles):
    """
    Set vel attributes for a set of ions to zero.
    """
    N = len(particles)
    for i in range(0, N):
        particles[i].vel = np.zeros(3)
    return particles

def explode2(allmolecules, epsilon, tf=np.array([0.00000001])):
    """
    Takes a list of molecule objects. These have an attribute self.ions,
    which is a list of ions they split into after a Coulomb explosion.
    Function finds final positions at tf of each ion by integration (solve_equations call)
    then calculates their time of flight down an arbitrary time-of-flight mass spectrometer,
    and updates z positions to do so.
    x,y,z,t,m,q,vx,vy,vz for each put into a list, returns these lists.
    Parameters
    ----------
    - allmolecules: list of molecule objects. Must have already called their methods self.vib()
      and self.createFragments() so that each has a list of ion positions and charges where those
      positions are the 'vibrated' positions.
    - epsilon: detection efficiency parameter, between 0 and 1. Each ion has a chance of not being
      detected and therefore not put in the output lists.
    - tf: final time to evaluate positions. Check the code before updating this.

    Returns
    --------
    - xs, ys, zs, ts, ms, qs, vxs, vys, vzs: lists of final values of each of these for each ion in each molecule.
    """
    rannumgen = default_rng()
    xs = []
    ys = []
    zs = []
    ts = []
    ms = []
    qs = []
    vxs = []
    vys = []
    vzs = []
    for m in allmolecules:
        ion_list = m.ions
        N = len(ion_list)
        ion_list = init_vel(ion_list)
        ion_list = set_vel_elements(ion_list)
        ion_list = angstroms_to_metres(ion_list)
        charged = []
        for i in range(0, N):
            if ion_list[i].q == 0:
                continue
            else:
                charged.append(ion_list[i])
        N = len(charged)
        if N == 0:
            continue
        ions = charged
        seed = rannumgen.integers(100000)
        ions = rotate_molecule(ions, randstate=seed)
        t_span = (0.0,tf)
        sol = solve_equations(ions, t_to_return=np.array([0.00000001]), t_span=(0.0,tf))
        final_positions, final_velocities = np.split(sol.y, 2)
        #creating a way to iterate over three variables (x1, y1, z1...)
        final_positions = list(itertools.chain(*final_positions))
        final_velocities = list(itertools.chain(*final_velocities))
        final_positions = get_listofarrays(final_positions)
        final_velocities = get_listofarrays(final_velocities)
        for i in range(0,N):
            position_vector = final_positions[i]
            ions[i].pos = position_vector
        for i in range(0,N):
            velocity_vector = final_velocities[i]
            ions[i].vel = velocity_vector
            ions[i].vx = ions[i].vel[0]
            ions[i].vy = ions[i].vel[1]
            ions[i].vz = ions[i].vel[2]
        for i in range(0, N):
            ions[i].t = calc_time_of_flight(ions[i].q, ions[i].mass)
            final_pos = ions[i].vel*(ions[i].t - tf) + ions[i].pos
            ions[i].pos = final_pos
            # Uncomment the above 3 lines if you wish to calculate time of flights accurately and adjust the voltages in the
            # calc_time_of_flight function.
            # Otherwise the below line uses ion mass as a substitute for TOF, if youre only interested in spatial distribution.
            # ions[i].t = ions[i].mass
            chance = float(np.random.randint(0, 100+1))/100
            if chance > epsilon:
                # print('not detected')
                continue
            else:
                xs.append(ions[i].pos[0])
                ys.append(ions[i].pos[1])
                zs.append(ions[i].pos[2])
                ts.append(ions[i].t)
                ms.append(ions[i].mass)
                qs.append(ions[i].q)
                vxs.append(ions[i].vx)
                vys.append(ions[i].vy)
                vzs.append(ions[i].vz)
    return xs, ys, zs, ts, ms, qs, vxs, vys, vzs

