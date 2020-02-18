import torch
import torch.utils.data as Data
from scipy.linalg import eigvalsh
import training
import useful_utils
import numpy as np


class DataPropGeneration:
    def __init__(
        self,
        n=10,
        m=15,
        s=2,
        lambd=1.0,
        N_TRAIN=int(1e4),
        N_TEST=int(1e3),
        BATCH_SIZE=512,
        SCENARIO_FLAG=0,
        ORACLE_LISTA_FLAG=True,
        snr_dict=0,
        INPAINTING_RATIO=0.5,
        snr_signals=np.inf,
    ):
        self.n = n
        self.m = m
        self.s = s
        self.lambd = lambd
        self.N_TRAIN = N_TRAIN
        self.N_TEST = N_TEST
        self.BATCH_SIZE = BATCH_SIZE
        self.SCENARIO_FLAG = SCENARIO_FLAG
        self.ORACLE_LISTA_FLAG = ORACLE_LISTA_FLAG
        self.snr_dict = snr_dict
        self.INPAINTING_RATIO = INPAINTING_RATIO
        self.snr_signals = snr_signals
        # Print scenario information
        if not np.isinf(snr_signals):
            print(f"Signal-SNR {self.snr_signals}[dB]")
        if SCENARIO_FLAG == 1:
            print(f"Dictionary SNR = {self.snr_dict}[dB]")

    def create_data(self, training_data_dict={}):
        """Create train and test data for training.
        """
        useful_utils.print_section_seperator("Generating Data")

        # Dictionary Initialization
        D = torch.randn(self.n, self.m)
        D /= torch.norm(D, dim=0)
        training_data = training.TrainingData(
            n=self.n,
            m=self.m,
            s=self.s,
            D=D,
            lambd=self.lambd,
            SCENARIO_FLAG=self.SCENARIO_FLAG,
            ORACLE_LISTA_FLAG=self.ORACLE_LISTA_FLAG,
            snr_dict=self.snr_dict,
            inpainting_ratio=self.INPAINTING_RATIO,
            snr_signals=self.snr_signals,
            **training_data_dict,
        )

        # Create a data generator function
        if self.SCENARIO_FLAG == 0:  # Column Perturbations

            def generator_func(N, RAND_FLAG=True, D_noised=None, IS_TRAIN=True):
                return create_data_columns_perturbations(
                    training_data,
                    N=N,
                    BATCH_SIZE=self.BATCH_SIZE,
                    RAND_FLAG=RAND_FLAG,
                    snr_signals=self.snr_signals,
                )

        elif self.SCENARIO_FLAG == 1:  # Noisy Dictionary

            def generator_func(N, RAND_FLAG=True, D_noised=None, IS_TRAIN=True):
                return create_data_noised_dict(
                    training_data,
                    N=N,
                    BATCH_SIZE=self.BATCH_SIZE,
                    snr_dict=self.snr_dict,
                    RAND_FLAG=RAND_FLAG,
                    D_noised_given=D_noised,
                    snr_signals=self.snr_signals,
                )

        elif self.SCENARIO_FLAG == 2:  # Random Dictionary

            def generator_func(N, RAND_FLAG=True, D_noised=None, IS_TRAIN=True):
                return create_data_random_dict(
                    training_data,
                    N=N,
                    BATCH_SIZE=self.BATCH_SIZE,
                    RAND_FLAG=RAND_FLAG,
                    snr_signals=self.snr_signals,
                )

        elif self.SCENARIO_FLAG == 3:  # Image Inpainting

            def generator_func(N, IS_TRAIN=True):
                return create_data_inpainting(
                    training_data,
                    N=N,
                    BATCH_SIZE=self.BATCH_SIZE,
                    RATIO=self.INPAINTING_RATIO,
                    IS_TRAIN=IS_TRAIN,
                )

        training_data.generator_func = generator_func
        training_data.train_loader = generator_func(N=self.N_TRAIN, IS_TRAIN=True)
        training_data.test_loader = generator_func(N=self.N_TEST, IS_TRAIN=False)
        return training_data


# Scenario 0 -- Column Perturbations
def create_data_columns_perturbations(
    training_data, N=int(1e4), BATCH_SIZE=512, RAND_FLAG=True, snr_signals=np.inf
):
    """Create simulated data for the scenario of column perturbations, in which every signal is generated
    from a different perturbation of the dictioarny columns.

    Arguments:
        training_data {TrainingData} -- training data information

    Keyword Arguments:
        N {int} -- sample size (default: {int(1e4)})
        BATCH_SIZE {int} -- batch size (default: {512})
        RAND_FLAG {bool} -- random columns flag (False only in 'Oracle' case) (default: {True})
        snr_signals {float} -- snr of input signals (default: {np.inf})

    Returns:
        data_loader {Data.DataLoader} -- a DataLoader with simulated data
    """
    n, m, s, D = training_data.n, training_data.m, training_data.s, training_data.D
    # The maximal eigenvalue
    L = float(eigvalsh(D.t() @ D, eigvals=(m - 1, m - 1)))
    # Initialization
    y = torch.zeros(n, N)
    D_ind = torch.zeros(m, N, dtype=torch.long)
    x = torch.zeros(m, N)
    # Create signals
    for i in range(N):
        # Random columns indices
        D_ind[:, i] = torch.randperm(m) if RAND_FLAG else torch.arange(m)
        D_i = D[:, D_ind[:, i]]
        # Random s indices and values
        x_ind = torch.randperm(m)[:s]
        x_nz = torch.randn(s)
        # Create the signal
        y[:, i] = D_i[:, x_ind] @ x_nz
        # Add noise to the signal
        if not np.isinf(snr_signals):
            noise = snr2std(y[:, i], snr_signals) * torch.randn(n)
            y[:, i] += noise
        # Create the sparse representation by solving the Basis-Pursuit
        x[:, i] = fista(y=y[:, i], D=D_i, lambd=training_data.lambd, L=L)

    simulated = SimulatedDataPerturbations(y=y, D=D, D_ind=D_ind, x=x)
    data_loader = Data.DataLoader(
        dataset=simulated, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    return data_loader


# Scenario 1 -- Noisy Dictionary
def create_data_noised_dict(
    training_data,
    N=int(1e4),
    BATCH_SIZE=512,
    snr_dict=20,
    RAND_FLAG=True,
    D_noised_given=None,
    snr_signals=np.inf,
):
    """Create simulated data for the scenario of noisy dictionaries, in which every signal is generated
    from a different noisy dictionary.

    Arguments:
        training_data {TrainingData} -- training data information

    Keyword Arguments:
        N {int} -- sample size (default: {int(1e4)})
        BATCH_SIZE {int} -- batch size (default: {512})
        snr_dict {float} -- snr of the dictionaries
        RAND_FLAG {bool} -- random noisy dictionary (False only in 'Oracle' case) (default: {True})
        D_noised_given {torch} -- a given and fixed dictionary for the 'Oracle' scenario
        snr_signals {float} -- snr of input signals (default: {np.inf})

    Returns:
        data_loader {Data.DataLoader} -- a DataLoader with simulated data
    """
    n, m, s, D = training_data.n, training_data.m, training_data.s, training_data.D
    # The maximal eigenvalue
    L = float(eigvalsh(D.t() @ D, eigvals=(m - 1, m - 1)))
    # Initialization
    D_db = 10 * torch.log10(D.var())
    noise_db = D_db - snr_dict
    noise_std = 10 ** (noise_db / 20)
    if D_noised_given is None:  # Oracle case
        D_noised = torch.zeros(n, m, N)
    else:
        if RAND_FLAG:
            D_noised = D_noised_given.clone()
        else:
            D_noised = D.unsqueeze(2).expand(-1, -1, N)
    y = torch.zeros(n, N)
    x = torch.zeros(m, N)
    # Create signals
    for i in range(N):
        if D_noised_given is None:
            # Random columns indices
            noise = noise_std * torch.randn(n, m) if RAND_FLAG else 0
            D_noised[:, :, i] = D + noise
        # Random s indices and values
        x_ind = torch.randperm(m)[:s]
        x_nz = torch.randn(s)
        # Create the signal
        y[:, i] = D_noised[:, x_ind, i] @ x_nz
        # Add noise to the signal
        if not np.isinf(snr_signals):
            noise = snr2std(y[:, i], snr_signals) * torch.randn(n)
            y[:, i] += noise
        # Find the sparse representation by solving the Basis-Pursuit
        x[:, i] = fista(y=y[:, i], D=D_noised[:, :, i], lambd=training_data.lambd, L=L)

    simulated = SimulatedDataNoisedDict(y=y, D_noised=D_noised, x=x)
    data_loader = Data.DataLoader(
        dataset=simulated, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    return data_loader


# Scenario 2 -- Random Dictionary
def create_data_random_dict(
    training_data, N=int(1e4), BATCH_SIZE=512, RAND_FLAG=True, snr_signals=np.inf
):
    """Create simulated data for the scenario of random dictionaries, in which every signal is generated
    from a different random dictionary.

    Arguments:
        training_data {TrainingData} -- training data information

    Keyword Arguments:
        N {int} -- sample size (default: {int(1e4)})
        BATCH_SIZE {int} -- batch size (default: {512})
        RAND_FLAG {bool} -- random dictionary flag (False only in 'Oracle' case) (default: {True})
        snr_signals {float} -- snr of input signals (default: {np.inf})

    Returns:
        data_loader {Data.DataLoader} -- a DataLoader with simulated data
    """
    n, m, s, D = training_data.n, training_data.m, training_data.s, training_data.D
    # The maximal eigenvalue
    L = float(eigvalsh(D.t() @ D, eigvals=(m - 1, m - 1)))
    L *= 5  # as to randomness
    # Initialization
    y = torch.zeros(n, N)
    D_noise = torch.zeros(n, m, N)
    x = torch.zeros(m, N)
    # Create signals
    for i in range(N):
        # Random dictionary
        if RAND_FLAG:
            D_i = torch.randn(n, m)
            D_i /= torch.norm(D_i, dim=0)
        else:
            D_i = D
        D_noise[:, :, i] = D_i
        # Random s indices and values
        x_ind = torch.randperm(m)[:s]
        x_nz = torch.randn(s)
        # Create the signal
        y[:, i] = D_noise[:, x_ind, i] @ x_nz
        # Add noise to the signal
        if not np.isinf(snr_signals):
            noise = snr2std(y[:, i], snr_signals) * torch.randn(n)
            y[:, i] += noise
        # Find the sparse representation by solving the Basis-Pursuit
        x[:, i] = fista(y=y[:, i], D=D_noise[:, :, i], lambd=training_data.lambd, L=L)

    simulated = SimulatedDataNoisedDict(y=y, D_noised=D_noise, x=x)
    data_loader = Data.DataLoader(
        dataset=simulated, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    return data_loader


# Scenario 3 -- Image Inpainting
def create_data_inpainting(training_data, N=int(1e4), BATCH_SIZE=512, RATIO=0.5, IS_TRAIN=True):
    """Create simulated data for the scenario of random dictionaries, in which every signal is generated
    from a different random dictionary.

    Arguments:
        training_data {TrainingData} -- training data information

    Keyword Arguments:
        N {int} -- sample size (default: {int(1e4)})
        BATCH_SIZE {int} -- batch size (default: {512})
        RATIO {float} - percent of missing pixels (default: {0.5})
        IS_TRAIN {bool} -- training or validation data flag (default: {True})

    Returns:
        data_loader {Data.DataLoader} -- a DataLoader with simulated data
    """
    n, m = training_data.n, training_data.m

    # load existing patches and dictionary or create new ones
    y_all, D, _, _ = useful_utils.collect_patches_and_dict(
        patch_size=int(np.sqrt(n)), num_atoms=m, num_patches_train=N
    )
    y_clean = (
        torch.from_numpy(y_all["train"].astype(np.float32))
        if IS_TRAIN
        else torch.from_numpy(y_all["val"].astype(np.float32))
    )
    y_clean = y_clean[:, torch.randperm(y_clean.shape[1])[:N]]
    D = torch.from_numpy(D.astype(np.float32))
    # update D in training data to contain loaded dictionary
    training_data.D = D
    N = y_clean.shape[1]
    # The maximal eigenvalue
    L = float(eigvalsh(D.t() @ D, eigvals=(m - 1, m - 1)))
    # Inpainting mask
    M = torch.where(torch.rand(n, N) < RATIO, torch.zeros(n, N), torch.ones(n, N))
    y = M * y_clean
    # turn M into diagonal form
    M_tensor = (torch.diag_embed(M.T)).permute(1, 2, 0)
    x = fista_inpainting(y=y, D=D, M=M_tensor, lambd=training_data.lambd, L=L, max_itr=300)
    # x = fista(y=y_clean, D=D, lambd=training_data.lambd, L=L)

    simulated = SimulatedDataNoisedDict(y=y, D_noised=M_tensor, x=x)
    data_loader = Data.DataLoader(
        dataset=simulated, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    return data_loader


def ista(y, D, lambd=1.0, L=None, max_itr=100):
    """ISTA Solver.

    Arguments:
        y {torch} -- signal
        D {torch} -- dictionary
        lambda {float} -- lagrangian multiplier

    Keyword Arguments:
        L {float} -- maximal eigenvalue (default: {None})
        max_itr {int} -- maximum iterations
    """
    if L is None:
        m = D.shape[1]
        Gram = torch.matmul(D.t(), D)
        L = float(eigvalsh(Gram, eigvals=(m - 1, m - 1)))
    x = torch.zeros(D.shape[1])
    proj = torch.nn.Softshrink(lambd=lambd / L)
    for _ in range(max_itr):
        x_tild = x - 1 / L * (D.T @ (D @ x - y))
        x = proj(x_tild)
    return x


def fista(y, D, lambd=1.0, L=None, max_itr=100):
    """FISTA Solver.

    Arguments:
        y {torch} -- signal
        D {torch} -- dictionary
        lambda {float} -- lagrangian multiplier

    Keyword Arguments:
        L {float} -- maximal eigenvalue (default: {None})
        max_itr {int} -- maximum iterations
    """
    matrix_form = True if len(y.shape) == 2 else False

    if L is None:
        m = D.shape[1]
        Gram = torch.matmul(D.t(), D)
        L = float(eigvalsh(Gram, eigvals=(m - 1, m - 1)))
    t_curr = 1
    if not matrix_form:
        x_curr = torch.zeros(D.shape[1])
        z = torch.zeros(D.shape[1])
        proj = torch.nn.Softshrink(lambd=lambd / L)
    else:
        x_curr = torch.zeros(D.shape[1], y.shape[1])
        z = torch.zeros(D.shape[1], y.shape[1])
        thresh = lambd / L

        def proj(x):
            return torch.nn.functional.relu(x.abs() - thresh) * x.sign()

    for _ in range(max_itr):
        t_prev = t_curr
        t_curr = 0.5 * (1 + np.sqrt(1 + 4 * (t_prev ** 2)))

        x_prev = x_curr
        z_tild = z - 1 / L * (D.T @ (D @ z - y))
        x_curr = proj(z_tild)

        z = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)
    return x_curr


def ista_inpainting(y, D, M, lambd=1.0, L=None, max_itr=100, same_L=True):
    """ISTA Solver.

    Arguments:
        y {torch} -- signal
        D {torch} -- dictionary
        lambda {float} -- lagrangian multiplier

    Keyword Arguments:
        L {float} -- maximal eigenvalue (default: {None})
        max_itr {int} -- maximum iterations
    """
    if L is None:
        m = D.shape[1]
        if same_L:
            Gram = torch.matmul(D.t(), D)
            L = float(eigvalsh(Gram, eigvals=(m - 1, m - 1)))
        else:
            L = torch.zeros(M.shape[-1])
            for i in range(M.shape[-1]):
                D_tild = M[:, :, i] @ D
                Gram = D_tild.T @ D_tild
                L[i] = float(eigvalsh(Gram, eigvals=(m - 1, m - 1)))

    M_mat = torch.diagonal(M).T
    y_term = (1 / L) * (D.T @ (M_mat * y))
    x = torch.zeros(D.shape[1], y.shape[1])
    thresh = lambd / L

    def proj(x):
        return torch.nn.functional.relu(x.abs() - thresh) * x.sign()

    for _ in range(max_itr):
        x_tild = x + y_term - (1 / L) * (D.T @ (M_mat * (D @ x)))
        x = proj(x_tild)
    return x


def fista_inpainting(y, D, M, lambd=1.0, L=None, max_itr=100, same_L=True):
    """FISTA Solver.

    Arguments:
        y {torch} -- signal
        D {torch} -- dictionary
        lambda {float} -- lagrangian multiplier

    Keyword Arguments:
        L {float} -- maximal eigenvalue (default: {None})
        max_itr {int} -- maximum iterations
    """

    if L is None:
        m = D.shape[1]
        if same_L:
            Gram = torch.matmul(D.t(), D)
            L = float(eigvalsh(Gram, eigvals=(m - 1, m - 1)))
        else:
            L = torch.zeros(M.shape[-1])
            for i in range(M.shape[-1]):
                D_tild = M[:, :, i] @ D
                Gram = D_tild.T @ D_tild
                L[i] = float(eigvalsh(Gram, eigvals=(m - 1, m - 1)))

    M_mat = torch.diagonal(M).T
    y_term = (1 / L) * (D.T @ (M_mat * y))
    thresh = lambd / L
    t_curr = 1
    x_curr = torch.zeros(D.shape[1], y.shape[1])
    z = torch.zeros(D.shape[1], y.shape[1])

    def proj(x):
        return torch.nn.functional.relu(x.abs() - thresh) * x.sign()

    for _ in range(max_itr):
        x_prev = x_curr
        z_tild = z + y_term - (1 / L) * (D.T @ (M_mat * (D @ z)))
        x_curr = proj(z_tild)

        t_prev = t_curr
        t_curr = 0.5 * (1 + np.sqrt(1 + 4 * (t_prev ** 2)))
        z = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)

    return x_curr


class SimulatedDataPerturbations(Data.Dataset):
    """Simulated dataset."""

    def __init__(self, y, D, D_ind, x):
        """Init.

        Arguments:
            y {torch} -- signals
            D {torch} -- dictionary
            D_ind {torch} -- perturbations indices
            x {torch} -- latent space (representations)
        """
        self.y = y
        self.x = x
        self.D = D
        self.D_ind = D_ind

    def __len__(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        y = self.y[:, idx]
        D = self.D[:, self.D_ind[:, idx]]
        x = self.x[:, idx]
        return y, D, x


class SimulatedDataNoisedDict(Data.Dataset):
    """Simulated dataset."""

    def __init__(self, y, D_noised, x):
        """Init.

        Arguments:
            y {torch} -- signals
            D {torch} -- dictionary
            x {torch} -- latent space (representations)
        """
        self.y = y
        self.x = x
        self.D_noised = D_noised

    def __len__(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        y = self.y[:, idx]
        D = self.D_noised[:, :, idx]
        x = self.x[:, idx]
        return y, D, x


def snr2std(y, snr):
    s_db = 10 * torch.log10(y.var())
    noise_db = s_db - snr
    noise_std = 10 ** (noise_db / 20)
    return noise_std
