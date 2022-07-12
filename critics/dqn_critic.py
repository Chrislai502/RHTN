from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from cs285.infrastructure import pytorch_util as ptu


class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']
        
        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']

        #RHTN
        self.num_networks = hparams['num_networks']
        self.increment_steps = hparams['increment_steps']
        self.optimizer = []
        self.learning_rate_scheduler=[]
        self.networks = [network_initializer(self.ob_dim, self.ac_dim) for _ in range(self.num_networks)]
        self.optimizer = []
        self.learning_rate_scheduler=[]
        for i in range(self.num_networks):
            self.optimizer.append(self.optimizer_spec.constructor(
                self.networks[i].parameters(),
                **self.optimizer_spec.optim_kwargs
            ))
            self.networks[i].to(ptu.device)
            self.learning_rate_scheduler.append(optim.lr_scheduler.LambdaLR(
            self.optimizer[i],
            self.optimizer_spec.learning_rate_schedule,
            ))

        self.loss = nn.SmoothL1Loss()  # AKA Huber loss


    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n, horizon, network_now, network_next):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        # RHTN modified
        # optimizers = [self.optimizer_spec.constructor(
        #     x.parameters(),
        #     **self.optimizer_spec.optim_kwargs
        # ) for x in self.networks]

        # RHTN modified
        # question: how to fit/regress?  
        # currently doing repeated gradient descent on the same tuples of obs and acs
        if not self.switch:
            i = horizon
            if horizon == 0:
                #network_now = self.networks[0]
                # set equal to reward function
                ra_t_values = self.networks[i](ob_no) # network_next
                r_t_values = torch.gather(ra_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
                target = reward_n
                loss = self.loss(r_t_values, target)
            else:     
                qa_t_values = self.networks[i](ob_no) # network_next
                q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
                qa_tp1_values = self.networks[i-1](next_ob_no) # network_now
                if self.double_q:
                    # You must fill this part for Q2 of the Q-learning portion of the homework.
                    # In double Q-learning, the best action is selected using the Q-network that
                    # is being updated, but the Q-value for this action is obtained from the
                    # target Q-network. Please review Lecture 8 for more details,
                    # and page 4 of https://arxiv.org/pdf/1509.06461.pdf is also a good reference.
                    next_actions = self.q_net(next_ob_no).argmax(dim=1)
                    q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_tp1, _ = qa_tp1_values.max(dim=1)

                # TODO compute targets for minimizing Bellman error
                # HINT: as you saw in lecture, this would be:
                    #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
                target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
                target = target.detach()
                loss = self.loss(q_t_values, target)

                #RHTN
            self.optimizer[i].zero_grad()
            loss.backward()
            utils.clip_grad_value_(self.networks[i].parameters(), self.grad_norm_clipping)
            self.optimizer[i].step()
            self.learning_rate_scheduler[i].step()
        else:
            # if we use the switch-based method
            # build a new network_next 
            network_next = self.networks[1]
            qa_t_values = network_next(ob_no) # network_next
            q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
            qa_tp1_values = self.networks[0](next_ob_no) # network_now
            q_tp1, _ = qa_tp1_values.max(dim=1)

            target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
            target = target.detach()
            loss = self.loss(q_t_values, target)

            # build network_next optimizer
            if horizon == 0:
                self.optimizer = self.optimizer_spec.constructor(
                        network_next.parameters(),
                        **self.optimizer_spec.optim_kwargs
                    )
                self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                    self.optimizer_spec.learning_rate_schedule,
                    )

            # increment after increment_steps
            for _ in range(self.increment_steps):
                self.optimizer.zero_grad()
                loss.backward()
                utils.clip_grad_value_(network_next.parameters(), self.grad_norm_clipping)
                self.optimizer.step()
                self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        if self.switch:
            qa_values = self.networks[0](obs)
        else:
            qa_values = self.networks[-1](obs)
        return ptu.to_numpy(qa_values)
