import torch
import numpy as np
import random
from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
import pickle
from training import add_and_pad, pad_batch_online

import time

########################################################################################################################
###################################### Helper Functions ################################################################
########################################################################################################################

def plot_fake_mels(gen, fixed_noise,fixed_real, fixed_vector,
                   epoch_ii,plot_save_after_i_iterations,
                   fixed_files,
                   dict_file,file_to_store,
                   starting_i = 23):
    """
        Plot 6 generated log-mel spectrograms from a fixed noise Tensor in comparison to 6 fixed real examples

        :param gen: torch model (generator)
            generator model
        :param fixed_noise: 3D torch.Tensor (batch, 1, z_dim)
            Tensor containing a random noise vector for each example to generate from batch
        :param fixed_real: 3D torch.Tensor (batch, time, mel_channel)
            Tensor containing a batch of real examples
        :param fixed_vector: 2D torch.Tensor (batch, sem_vec)
            Tensor containing the corresponding semantic vector for each sample in fixed_real
        :param epoch_ii: int
            current epoch in training
        :param plot_save_after_i_iterations: int
            number of epochs to train model before saving generated examples
        :param fixed_files: pd.Series
            corresponding file to each real example
        :param dict_file: str
            name of the dictionary to save the plot in
        :param file_to_store: str
            file name for saving the plot
        :param starting_i: int
            index in batch at which we start to take 6 succesive examples (e.g. ensure to take 3 recorded and 3 synthesized examples)
        :return: None
        """

    with torch.no_grad():
        fake = gen(fixed_noise, len(fixed_real[0]), fixed_vector)
        fig, axes = plt.subplots(nrows=4, ncols=3, facecolor="white", figsize=(15, 10), sharey=True)
        i = starting_i  # e.g. ensure 3 recorded 3 synthesized examples
        for row in range(2):
            for col in range(3):
                # if i < 8:
                ax1 = axes[row * 2, col]
                ax2 = axes[row * 2 + 1, col]

                real_img = fixed_real[i].detach().cpu().T
                fake_img = fake[i].detach().cpu().T
                ax1.imshow(real_img, origin='lower')
                ax2.imshow(fake_img, origin='lower')

                if not fixed_files is None:
                    ax1.set_title(fixed_files.iloc[i], fontsize=18, pad=5)
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax1.yaxis.tick_right()
                ax2.yaxis.tick_right()

                if col == 2:
                    ax1.yaxis.set_label_position("right")
                    ax1.set_ylabel('Mel Channel', fontsize=15, rotation=270, labelpad=20)
                    ax2.yaxis.set_label_position("right")
                    ax2.set_ylabel('Mel Channel', fontsize=15, rotation=270, labelpad=20)

                else:
                    ax1.set_yticks([])
                    ax2.set_yticks([])

                i += 1
        #fig.subplots_adjust(hspace=-0.03)
        axes[0, 0].text(-0.1, 0.5, "Real", fontsize=18, va="center", rotation=90,
                        transform=axes[0, 0].transAxes)
        axes[1, 0].text(-0.1, 0.5, "Fake", fontsize=18, va="center", rotation=90,
                        transform=axes[1, 0].transAxes)
        axes[2, 0].text(-0.1, 0.5, "Real", fontsize=18, va="center", rotation=90,
                        transform=axes[2, 0].transAxes)
        axes[3, 0].text(-0.1, 0.5, "Fake", fontsize=18, va="center", rotation=90,
                        transform=axes[3, 0].transAxes)
        fig.subplots_adjust(hspace=0.2, wspace=-0.0
                            )


        if (epoch_ii + 1) % plot_save_after_i_iterations == 0:
            plt.savefig(dict_file + file_to_store % (epoch_ii + 1) + ".jpg")
        plt.show()

def plot_fake_cps(gen, fixed_noise,fixed_real, fixed_vector,
                   epoch_ii,plot_save_after_i_iterations,
                   fixed_files,
                   dict_file,file_to_store, colors, n_cps = 5):
    """
            Plot the first 5 cp-trajectories from 9 different words in comparison with the same cps generated from a fixed noise Tensor

            :param gen: torch model (generator)
                generator model
            :param fixed_noise: 3D torch.Tensor (batch, 1, z_dim)
                Tensor containing a random noise vector for each example to generate from batch
            :param fixed_real: 3D torch.Tensor (batch, time, cp)
                Tensor containing a batch of real examples
            :param fixed_vector: 2D torch.Tensor (batch, sem_vec)
                Tensor containing the corresponding semantic vector for each sample in fixed_real
            :param epoch_ii: int
                current epoch in training
            :param plot_save_after_i_iterations: int
                number of epochs to train model before saving generated examples
            :param fixed_files: pd.Series
                corresponding file to each real example
            :param dict_file: str
                name of the dictionary to save the plot in
            :param file_to_store: str
                file name for saving the plot
            :param colors: list
                list of colors for different cps
            :param n_cps: int (default = 5)
                number of cps to plot
            :return: None
            """
    assert len(colors) <= n_cps, "Not enough colors provide for plotting distinct cp-trajectories: %d provided %d needed!" % (len(colors), n_cps)
    with torch.no_grad():
        fake = gen(fixed_noise,len(fixed_real[0]),fixed_vector)
        fig, axes = plt.subplots(nrows = 3, ncols=3, facecolor = "white", figsize = (15,10))
        i = 0
        for row in range(3):
            for col in range(3):
                ax = axes[row,col]
                for c in range(n_cps):
                    ax.plot(fake[i][:,c].detach().cpu(),color = colors[c])
                    ax.plot(fixed_real[i][:,c].detach().cpu(), color = colors[c], linestyle = "dotted")
                    if row < 2:
                        ax.set_xticks([])
                    if col > 0:
                        ax.set_yticks([])
                ax.set_title(fixed_files.iloc[i],fontsize=18, pad=7)
                ax.set_ylim((-1.1,1.1))
                i+=1
        fig.subplots_adjust(hspace = 0.2, wspace = 0.1)

        if (epoch_ii + 1) % plot_save_after_i_iterations == 0:
            plt.savefig(dict_file + file_to_store % (epoch_ii + 1) + ".jpg")
        plt.show()



########################################################################################################################
################################################ Training ##############################################################
########################################################################################################################

class Training:
    """
        Create Training Instance
        :param gen: torch model
            generator model to train
        :param critic: torch model
            critic model to train
        :param seed: int or str
            int to set random.seed in order to be reproducible
        :param torch_seed: int or str
            int to set torch.manual_seed in order to be reproducible
        :param target_name: str (one of ["mel", "cp"]) nedded to call correct plotting function
        :param inps: pd.Series
            series containing the inputs of the training set
        :param vectors: pd.Series
            series containing the corresponding semantic vectors
        :parram z_dim: int
            number of noise dimensions
        :param batch_size: int
            batch size
        :param res_train: pd.DataFrame
            pd.DataFrame for logging epoch results on training set

        :param opt_gen: torch.optim.Optimizer
            torch optimizer for updating weights of the generator model
        :param opt_critic: torch.optim.Optimizer
            torch optimizer for updating weights of the critic model

        :param use_same_size_batching: bool
            specify whether to batch inps with similar length during epoch creating in order to avoid long padding

        :param files: pd.Series
            corresponding file to each real example

        """
    def __init__(self, gen, critic, seed,torch_seed,tgt_name, inps, vectors,z_dim,
                 batch_size, res_train , opt_gen, opt_critic,use_same_size_batching = False,
                 files = None):

        self.seed = seed
        self.torch_seed = torch_seed
        self.gen = gen
        self.critic = critic
        self.opt_gen = opt_gen
        self.opt_critic = opt_critic

        self.device = next(gen.parameters()).device

        assert tgt_name in ["mel", "cp"], "Please provide a valid tgt name (mel or cp)!"
        self.tgt_name = tgt_name
        self.inps = inps
        self.vectors = torch.Tensor(np.array(list(vectors))).double().to(self.device)
        # get lengths of inputs and outputs
        self.lens = torch.Tensor(np.array(inps.apply(lambda x: len(x)), dtype=np.int)).to(self.device)
        self.z_dim = z_dim

        self.files = files

        self.batch_size = batch_size

        self.res_train = res_train
        self.res_train_ix = len(res_train)

        # is using same size batching we create a dictionary containing all unique lengths and the indices of each sample with a this length
        if use_same_size_batching:
            self.train_length_dict = {}
            lengths, counts = np.unique(self.lens.cpu(), return_counts=True)
            self.sorted_length_keys = np.sort(lengths)

            for length in self.sorted_length_keys:
                self.train_length_dict[length] = np.where(self.lens.cpu() == length)[0]

        # set seed and create fixed random noise vectors and a fixed batch of real examples to monitor training
        random.seed(self.seed)
        torch.manual_seed(self.torch_seed)
        self.fixed_batch = self.create_epoch_batches(len(self.inps), batch_size=self.batch_size, same_size_batching=True)[0]
        self.fixed_noise = torch.randn(self.batch_size, 1, self.z_dim).to(self.device)
        self.fixed_vector = self.vectors[self.fixed_batch]
        self.fixed_len = self.lens[self.fixed_batch]

        self.fixed_real = self.inps.iloc[self.fixed_batch]
        self.fixed_real = pad_batch_online(self.fixed_len, self.fixed_real, self.device).to(self.device)

        if not files is None:
            self.fixed_files = self.files.iloc[self.fixed_batch]
        else:
            self.fixed_files = None


    def gradient_penalty(self, critic, lens, vectors, real, fake, device="cpu"):
        """
        Gradient Penalty to enforce the Lipschitz constraint in order for the critic to be able to approximate a valid
        1-Lipschitz function. This is needed in order to use the Kantorovich-Rubinstein duality for simplifying the
        calculation of the wasserstein distance

        :param critic: torch model
            critic model
        :param lens: 2D torch.Tensor
            Tensor containing the real unpadded length of each sample in batch (batch, length)
        :param vectors: 2D torch.Tensor
            Tensor containing the corresponding semantic vectors
        :param real: 3D torch.Tensor (batch, time, cps)
            Tensor containing real images
        :param fake: 3D torch.Tensor (batch, time, cps)
            Tensor containing generated images
        :param device: str
            device to run calculation on
        :return:
        """

        batch_size, length, c = real.shape
        alpha = torch.rand((batch_size, 1, 1)).repeat(1, length, c).to(device)

        interpolated_input = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = critic(interpolated_input, lens, vectors)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_input,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.contiguous().view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def create_epoch_batches(self, df_length, batch_size, shuffle=True, same_size_batching=False):
        """
        :param df_length: int
            total number of samples in training set
        :param batch_size: int
            number of samples in one atch
        :param shuffle:
            keep order of training set or random shuffle
        :return epoch: list of list
            list of listis containing indices for each batch for one epoch
        """
        if same_size_batching:
            epoch = []
            foundlings = []
            for length in self.sorted_length_keys:
                length_idxs = self.train_length_dict[length]
                rest = len(length_idxs) % batch_size
                random.shuffle(length_idxs)
                epoch += [length_idxs[i * batch_size:(i * batch_size) + batch_size] for i in
                          range(int(len(length_idxs) / batch_size))]
                if rest > 0:
                    foundlings += list(length_idxs[-rest:])
            foundlings = np.asarray(foundlings)
            rest = len(foundlings) % batch_size
            epoch += [foundlings[i * batch_size:(i * batch_size) + batch_size] for i in
                      range(int(len(foundlings) / batch_size))]
            if rest > 0:
                epoch += [foundlings[-rest:]]
            random.shuffle(epoch)

        else:
            rest = df_length % batch_size
            idxs = list(range(df_length))
            if shuffle:
                random.shuffle(idxs)
            if rest > 0:
                idxs += idxs[:(batch_size - rest)]
            epoch = [idxs[i * batch_size:(i * batch_size) + batch_size] for i in range(int(len(idxs) / batch_size))]

        return epoch


    def train(self, num_epochs,
              continue_training_from,
              critic_iteration_schedule,
              lambda_gp,
              plot_every_i_iterations=1,
              save_plot_after_i_iterations=1,
              save_model_after_i_iterations=1,
              shuffle=True,
              verbose=True,
              dict_file="",
              file_to_store= time.strftime("%Y%m%d-%H%M%S")):

        """
                Train the conditional gan models with Wasserstein Distance + Gradient Penalty

                :param num_epochs: int
                    number of epochs to train
                :param continue_training_from: int
                    epoch to resume training from
                :param critic_iteration_schedule: dict {n_critic_iterations : for_n_epochs}
                    dictionary containing the schedule of critic iterations
                :param lambda_gp: int
                    weighting hyperparameter for gradient penalty
                :param plot_every_i_iterations: int
                    number of epochs before plotting the results on the fixed batch
                :param save_plot_after_i_iterations: int
                    number of epochs before saving the plotted results
                :param save_model_after_i_iterations: int
                    number of epochs to train model before saving it
                :param shuffle: bool
                    whether to use shuffle in creating epochs
                :param verbose: bool
                    print results after each epoch
                :param dict_file: str
                    dictionary to store the model in
                :param file_to_store: str
                    name of files to store model, training and validation results
                """

        critic_iterations = list(critic_iteration_schedule.keys())
        critic_iterations_switch_points = np.cumsum(list(critic_iteration_schedule.values()))


        if continue_training_from > 0:
            print("Continue Training: iteration %d..." % continue_training_from)
            for i in range(continue_training_from):
                epoch = self.create_epoch_batches(len(self.inps),  self.batch_size,shuffle=shuffle,same_size_batching=self.use_same_size_batching)
                critic_iteration = critic_iterations[np.where(i < critic_iterations_switch_points)[0][0]]
                for jj, idxs in epoch:
                    for _ in range(critic_iteration):
                        cur_batch_size = len(idxs)
                        noise = torch.randn(cur_batch_size, 1, self.z_dim).to(self.device)
        else:
            print("Start Training... ")

        for ii in tqdm(range(num_epochs), desc='Training...', position=0, leave=True):
            ii += continue_training_from
            epoch = self.create_epoch_batches(len(self.inps), self.batch_size, shuffle=shuffle, same_size_batching=self.use_same_size_batching)

            total_loss_gen = []
            total_loss_critic_gp = []
            total_loss_critic_diff = []

            critic_iteration = critic_iterations[np.where(ii < critic_iterations_switch_points)[0][0]]

            for jj, idxs in enumerate(tqdm(epoch, desc="Batch...", position=1, leave=False)):

                lens_input_jj = self.lens[idxs]
                real = self.inps[idxs]
                real = pad_batch_online(lens_input_jj, real, self.device)
                vectors_jj = self.vectors[idxs]
                cur_batch_size = real.shape[0]
                cur_batch_length = real.shape[1]

                # Train Critic: max E[critic(real)] - E[critic(fake)]
                # equivalent to minimizing the negative of that
                # with torch.backends.cudnn.flags(enabled=False): #https://github.com/facebookresearch/higher#knownpossible-issues
                if critic_iteration > 0:
                    for _ in range(critic_iteration):
                        noise = torch.randn(cur_batch_size, 1, self.z_dim).to(self.device)
                        fake = self.gen(noise, cur_batch_length, vectors_jj)

                        critic_real = self.critic(real, lens_input_jj, vectors_jj).reshape(-1)
                        critic_fake = self.critic(fake, lens_input_jj, vectors_jj).reshape(-1)

                        critic_diff = critic_real - critic_fake
                        loss_diff = torch.mean(critic_diff)

                        # STD
                        # critic_diff_std = torch.std(critic_real) - torch.std(critic_fake)

                        gp = self.gradient_penalty(self.critic, lens_input_jj, vectors_jj, real, fake, device=self.device)
                        loss_critic = -loss_diff + lambda_gp * gp  # - critic_diff_std
                        # loss_critic = -(loss_real - loss_fake) + LAMBDA_GP * gp
                        self.opt_critic.zero_grad()

                        loss_critic.backward(retain_graph=True)
                        # loss_critic.backward()
                        self.opt_critic.step()

                    total_loss_critic_gp += [loss_critic.item()]
                    # total_loss_critic += [-(loss_real - loss_fake).item()]
                    total_loss_critic_diff += [loss_diff.item()]
                    # total_loss_critic_diff_std += [critic_diff_std.item()]
                    # total_loss_critic_real += [-loss_real.item()]
                    # total_loss_critic_fake += [loss_fake.item()]
                else:
                    noise = torch.randn(cur_batch_size, 1, self.z_dim).to(self.device)
                    fake = self.gen(noise, cur_batch_length, vectors_jj)

                    critic_real = self.critic(real, lens_input_jj, vectors_jj).reshape(-1)
                    critic_fake = self.critic(fake, lens_input_jj, vectors_jj).reshape(-1)

                    critic_diff = critic_real - critic_fake
                    loss_diff = torch.mean(critic_diff)

                    # STD
                    # critic_diff_std = torch.std(critic_real) - torch.std(critic_fake)

                    gp = self.gradient_penalty(self.critic, lens_input_jj, vectors_jj, real, fake, device=self.device)
                    loss_critic = -loss_diff + lambda_gp * gp  # - critic_diff_std
                    # loss_critic = -(loss_real - loss_fake) + LAMBDA_GP * gp
                    self.opt_critic.zero_grad()

                    loss_critic.backward(retain_graph=True) # no step

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]

                gen_fake = self.critic(fake, lens_input_jj, vectors_jj).reshape(-1)
                critic_real = self.critic(real, lens_input_jj, vectors_jj).reshape(-1)

                critic_diff = critic_real - gen_fake
                # critic_diff_std = torch.std(critic_real) - torch.std(gen_fake)

                # loss_gen = -torch.mean(gen_fake)
                loss_gen = torch.mean(critic_diff)  # + critic_diff_std
                self.opt_gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

                total_loss_gen += [loss_gen.item()]

                for p in self.critic.parameters(): # clear gradients to reduce storage
                    p.grad = None
                for p in self.gen.parameters():
                    p.grad = None

                if verbose:
                    if (jj + 1) % 50 == 0:
                        print(f"Epoch [{ii}/{num_epochs+continue_training_from}] Batch {jj + 1}/{len(epoch)}")

            mean_loss_critic_gp = np.mean(total_loss_critic_gp)
            mean_loss_gen = np.mean(total_loss_gen)
            mean_loss_critic_diff = np.mean(total_loss_critic_diff)

            self.res_train.loc[self.res_train_ix] = [ii, mean_loss_critic_gp, mean_loss_critic_diff, mean_loss_gen]
            self.res_train_ix += 1

            if verbose:
                print(f"Epoch [{ii}/{num_epochs+continue_training_from}]  \
                              Loss D + GP: {mean_loss_critic_gp:.4f},Loss D Diff: {mean_loss_critic_diff:.4f},  loss G: {mean_loss_gen:.4f}")

            if (ii+1) % plot_every_i_iterations == 0:
                if self.tgt_name == "mel":
                    plot_fake_mels(self.gen, self.fixed_noise, self.fixed_real, self.fixed_vector,
                                   ii,save_plot_after_i_iterations,
                                   self.fixed_files,
                                   dict_file,file_to_store,starting_i = 23)

                else:
                    colors = ["C%d" % i for i in range(5)]
                    plot_fake_cps(self.gen,self.fixed_noise, self.fixed_real, self.fixed_vector,
                                  ii, save_plot_after_i_iterations,
                                  self.fixed_files,
                                  dict_file, file_to_store, colors)

            if (ii + 1) % save_model_after_i_iterations == 0:
                self.res_train.to_pickle(
                    dict_file + "/res_train_" + file_to_store + "_%d" % (ii + 1) + ".pkl")
                with open(dict_file + "/generator_"+ file_to_store + "_%d" % (ii + 1) + ".pkl",
                          "wb") as pfile:
                    pickle.dump((self.gen, self.opt_gen), pfile)
                with open(dict_file + "/critic_" + file_to_store + "_%d" % (ii + 1) + ".pkl", "wb") as pfile:
                    pickle.dump((self.critic, self.opt_critic), pfile)