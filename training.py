import numpy as np
import time
import torch.nn.functional as F
import torch
import generating
import models
import useful_utils
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt


class TrainingData:
    def __init__(
        self,
        n=10,
        m=15,
        s=2,
        lambd=1.0,
        D=None,
        train_loader=None,
        test_loader=None,
        EPOCH=30,
        cudaopt=False,
        lr=0.005,
        momentum=0.9,
        weight_decay=0,
        step_size=7,
        gamma=0.1,
        model=None,
        T=0,
        file_name="",
        SAVE_FLAG=True,
        ORACLE_LISTA_FLAG=True,
        train_loader_lista=None,
        test_loader_lista=None,
        LISTA_RANDOM_DATA=False,
        SCENARIO_FLAG=0,
        snr_dict=None,
        generator_func=None,
        inpainting_ratio=0,
        SAVE_MODEL=False,
        snr_signals=np.inf,
    ):
        self.n = n
        self.m = m
        self.s = s
        self.lambd = lambd
        self.D = D
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.EPOCH = EPOCH
        self.cudaopt = cudaopt
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.model = model
        self.T = T
        self.file_name = file_name
        self.SAVE_FLAG = SAVE_FLAG
        self.ORACLE_LISTA_FLAG = ORACLE_LISTA_FLAG
        self.train_loader_lista = train_loader_lista
        self.test_loader_lista = test_loader_lista
        self.LISTA_RANDOM_DATA = LISTA_RANDOM_DATA
        self.SCENARIO_FLAG = SCENARIO_FLAG
        self.snr_dict = snr_dict
        self.generator_func = generator_func
        self.inpainting_ratio = inpainting_ratio
        self.SAVE_MODEL = SAVE_MODEL
        self.snr_signals = snr_signals

    def training_procedure(self):
        """Training the different architectures:
        * Non-learned solvers - ISTA & FISTA.
        * Oracle LISTA - Train LISTA on fixed data (without randomness).
        * LISTA - Train LISTA on random data.
        * Ada-LISTA - Train Ada-LISTA on random data.

        Returns:
            err_dict [dict] -- dictionary with all the errors
        """
        err_dict = {}

        # Error without training
        err_dict["ISTA"] = self.ista_err()
        err_dict["FISTA"] = self.ista_err(FISTA_FLAG=True)

        # Error with an Orcale LISTA
        if self.ORACLE_LISTA_FLAG:
            err_dict["Oracle-LISTA"] = self.lista_err()

        # Error with LISTA on random data
        if self.LISTA_RANDOM_DATA:
            err_dict["LISTA"] = self.lista_err(LISTA_RANDOM_DATA=True)

        # Error of Adaptive-ISTA
        err_dict["Ada-LISTA"] = self.adalista_err()

        return err_dict

    def training_net(self):
        """Train a network.

        Returns:
            loss_test {numpy} -- loss function values on test set
        """
        # Initialization
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        loss_train = np.zeros((self.EPOCH,))
        loss_test = np.zeros((self.EPOCH,))
        time_test = np.zeros((self.EPOCH,))
        t0 = time.perf_counter()
        EPOCH_PRINT_NUM = 10  # Print status every EPOCH_PRINT_NUM epochs
        # Main loop
        for epoch in range(self.EPOCH):
            self.model.train()
            train_loss = 0
            for step, (b_y, b_D, b_x) in enumerate(self.train_loader):
                if self.cudaopt:
                    b_y, b_D, b_x = b_y.cuda(), b_D.cuda(), b_x.cuda()
                x_hat = self.model(b_y, b_D)
                loss = F.mse_loss(x_hat, b_x, reduction="sum")

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                self.model.zero_grad()
                train_loss += loss.data.item()
            loss_train[epoch] = train_loss / len(self.train_loader.dataset)
            if scheduler is not None:
                scheduler.step()

            # testing
            self.model.eval()
            test_loss = 0
            for step, (b_y, b_D, b_x) in enumerate(self.test_loader):
                if self.cudaopt:
                    b_y, b_D, b_x = b_y.cuda(), b_D.cuda(), b_x.cuda()
                x_hat = self.model(b_y, b_D)
                test_loss += F.mse_loss(x_hat, b_x, reduction="sum").data.item()
            loss_test[epoch] = test_loss / len(self.test_loader.dataset)
            time_test[epoch] = time.perf_counter() - t0
            # Print
            if epoch % EPOCH_PRINT_NUM == 0:
                print(
                    "Epoch %d, Train loss %.8f, Test loss %.8f, time %.2f"
                    % (epoch, loss_train[epoch], loss_test[epoch], time_test[epoch])
                )
            # Re-init if stuck at nan
            if np.isnan(test_loss):
                self.model.reinit()
                if self.cudaopt:
                    self.model.cuda()
                useful_utils.print_section_seperator(
                    "Warning!!! %s: Test loss is nan. Reinitialization number %d"
                    % (self.model._get_name(), self.model.reinit_num),
                    subsec=False,
                )
                return self.training_net()
        # Save the model
        if self.SAVE_MODEL:
            torch.save(self.model, self.file_name)

        return loss_test

    def ista_err(self, FISTA_FLAG=False):
        """Solve the Basis-Pursuit using ISTA/FISTA for T unfoldings.

        Keyword Arguments:
            FISTA_FLAG {bool} -- if True use FISTA (default: {False})

        Returns:
            loss {torch.tensor} -- test loss
        """
        solver = generating.ista if not FISTA_FLAG else generating.fista

        T = self.T + 1
        D = self.D
        m = D.shape[1]
        L = float(eigvalsh(D.t() @ D, eigvals=(m - 1, m - 1)))

        loss = 0
        for step, (y, D, x) in enumerate(self.test_loader.dataset):
            D_eff = D if self.SCENARIO_FLAG != 3 else D @ self.D
            x_hat = solver(y=y, D=D_eff, lambd=self.lambd, L=L, max_itr=T)
            loss += F.mse_loss(x_hat, x, reduction="sum").data.item()

        return loss / len(self.test_loader.dataset)

    def adalista_err(self):
        """Train Ada-LISTA on dataset.

        Returns:
            loss {torch.tensor} -- test loss
        """
        if not (self.SCENARIO_FLAG == 3):
            adap_ista = models.Adaptive_ISTA(
                n=self.n, m=self.m, T=self.T, lambd=self.lambd  # , D=self.D
            )
            useful_utils.print_section_seperator("Ada-LISTA Training", subsec=True)
        else:  # Image Inpainting
            adap_ista = models.Adaptive_FISTA_Rev(
                n=self.n, m=self.m, D=self.D, T=self.T, lambd=self.lambd
            )
            useful_utils.print_section_seperator("Ada-LFISTA-Rev Training", subsec=True)
        if self.cudaopt:
            adap_ista.cuda()
        training_data_adalista = self.create_training_data_adalista()
        training_data_adalista.model = adap_ista
        training_data_adalista.file_name = "./saved_models_inpainting/adalista_T%d" % self.T
        loss_test = training_data_adalista.training_net()
        err_adap_ista = loss_test[-1]
        return err_adap_ista

    def lista_err(self, LISTA_RANDOM_DATA=False):
        """Train LISTA on dataset.

        Keyword Arguments:
            LISTA_RANDOM_DATA {bool} -- dataset from random models. False in Oracle case. (default: {False})

        Returns:
            loss {torch.tensor} -- test loss
        """
        oracle_msg = "" if LISTA_RANDOM_DATA else " -- Oracle"
        lista = models.LISTA(n=self.n, m=self.m, T=self.T, D=self.D)
        useful_utils.print_section_seperator("LISTA Training" + oracle_msg, subsec=True)
        if self.cudaopt:
            lista.cuda()
        training_data_lista = (
            self.create_training_data_lista()
            if not LISTA_RANDOM_DATA
            else self.create_training_data_adalista()
        )
        training_data_lista.model = lista
        training_data_lista.file_name = "./saved_models_inpainting/lista_T%dR%d" % (
            self.T,
            int(LISTA_RANDOM_DATA),
        )
        loss_test = training_data_lista.training_net()
        err_lista = loss_test[-1]
        return err_lista

    def create_data_oracle_lista(self):
        """Create data for the Oracle case (signals from fixed model)"""
        if not self.ORACLE_LISTA_FLAG:
            return
        useful_utils.print_section_seperator("Generating Data for Oracle LISTA", subsec=True)
        D_noised_train, D_noised_test = None, None
        self.train_loader_lista = self.generator_func(
            N=len(self.train_loader.dataset), RAND_FLAG=False, D_noised=D_noised_train
        )
        self.test_loader_lista = self.generator_func(
            N=len(self.test_loader.dataset), RAND_FLAG=False, D_noised=D_noised_test
        )

    def create_training_data_lista(self):
        """Set the train and test dataloader to be the LISTA dataloaders."""
        return TrainingData(
            n=self.n,
            m=self.m,
            s=self.s,
            lambd=self.lambd,
            D=self.D,
            train_loader=self.train_loader_lista,
            test_loader=self.test_loader_lista,
            EPOCH=self.EPOCH,
            cudaopt=self.cudaopt,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            step_size=self.step_size,
            gamma=self.gamma,
            model=None,
            SAVE_FLAG=False,
            SAVE_MODEL=self.SAVE_MODEL,
        )

    def create_training_data_adalista(self):
        """Copy the relevant train and test dataloaders to TrainingData instance."""
        return TrainingData(
            n=self.n,
            m=self.m,
            s=self.s,
            lambd=self.lambd,
            D=self.D,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            EPOCH=self.EPOCH,
            cudaopt=self.cudaopt,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            step_size=self.step_size,
            gamma=self.gamma,
            model=None,
            SAVE_FLAG=False,
            SAVE_MODEL=self.SAVE_MODEL,
        )

    def plot_figure(self, T_opt, err_list, design_dict=None):
        """Plot the error results.

        Arguments:
            T_opt {list} -- the number of unfoldings
            err_list {list} -- list of test set errors for the different solvers.

        Keyword Arguments:
            design_dict {dict} -- plots design parameters (default: {None}).
        """
        # Create dict from the error list
        err_dict = {}
        for model_name in err_list[0]:
            err_dict[model_name] = np.array([err_list[i][model_name] for i in range(len(err_list))])
        # Plot the results
        fig = plt.figure()
        for model_name in err_dict:
            if design_dict is None:
                plt.plot(T_opt, err_dict[model_name], label=model_name)
            else:
                plt.plot(
                    T_opt,
                    err_dict[model_name],
                    label=model_name,
                    color=design_dict[model_name][0],
                    marker=design_dict[model_name][1],
                    fillstyle="none",
                )
        # Figure parameters
        plt.legend(loc="lower left")  # , **csfont)
        SCENARIO_DICT = {
            0: "Column Perturbations",
            1: "Noisy Dictionary",
            2: "Random Dictionary",
            3: "Image Inpainting",
        }
        title_msg = SCENARIO_DICT[self.SCENARIO_FLAG]
        if self.SCENARIO_FLAG == 1:
            title_msg += ", STD = %d" % self.snr_dict
        if self.SCENARIO_FLAG == 2:
            title_msg += ", Cardinality = %d" % self.s
        if self.SCENARIO_FLAG == 3:
            title_msg += ", Ratio = %.2f" % self.inpainting_ratio
        plt.title(title_msg)
        plt.xlabel("# iterations")
        plt.yscale("log")
        # Save the figures
        if self.SAVE_FLAG:
            if self.SCENARIO_FLAG == 0:
                experiment_title = "_columns_pert"
            elif self.SCENARIO_FLAG == 1:
                experiment_title = f"_noised_dict_snr{self.snr_dict}"
            elif self.SCENARIO_FLAG == 2:
                experiment_title = f"_random_dict_s{self.s}"
            elif self.SCENARIO_FLAG == 3:
                experiment_title = "_inpainting"
            if not np.isinf(self.snr_signals):
                experiment_title += f"_snr_{self.snr_signals}"
            fig_name = "ista_vs_adap_ista" + experiment_title
            fig.savefig("./figures/" + fig_name + ".pdf", bbox_inches="tight")
        plt.show()
        return
