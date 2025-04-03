import itertools
from typing import Dict, Optional, Type

import numpy as np
import torch

from research.networks.base import ActorCriticValuePolicy

from .off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


class RRD_IQL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticValuePolicy)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.clip_score = clip_score
        self.r_var = 1
        self.r_square = 1
        self.alpha = 1
        self.gamma = 1
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(                                                                    
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        keys = ("actor", "critic", "value")
        default_kwargs = {}
        for key, value in self.optim_kwargs.items():
            if key not in keys:
                default_kwargs[key] = value
            else:
                assert isinstance(value, dict), "Special keys must be kwarg dicts"

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = self.optim_class(actor_params, **actor_kwargs)

        # Update the encoder with the critic.
        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
        self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **critic_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(self.optim_kwargs.get("value", dict()))
        self.optim["value"] = self.optim_class(self.network.value.parameters(), **value_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        # Run the encoders



        obs = self.network.encoder(batch["obs"])

        with torch.no_grad():
            next_obs = self.network.encoder(batch["next_obs"])
        action = batch["action"]
  
        # compute the value loss
        with torch.no_grad():
            target_q = self.target_network.critic(obs, action)
            target_q = torch.min(target_q, dim=0)[0]
        vs = self.network.value(obs)
        v_loss = iql_loss(vs, target_q.expand(vs.shape[0], -1), self.expectile).mean()

        self.optim["value"].zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim["value"].step()

        # Next, update the actor. We detach and use the old value, v for computational efficiency
        # and use the target_q value though the JAX IQL recomputes both
        # Pytorch IQL versions have not.
        with torch.no_grad():
            adv = target_q - torch.mean(vs, dim=0)  # min trick is not used on value.
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.network.actor(obs)  # Use encoder gradients for the actor.
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(action).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape == action.shape
            bc_loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        actor_loss = (exp_adv * bc_loss).mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        # Next, Finally update the critic
        with torch.no_grad():
            next_vs = self.network.value(next_obs)
            next_v = torch.mean(next_vs, dim=0, keepdim=True)  # Min trick is not used on value.
            target = batch["reward"] + batch["discount"] * next_v  # use the predicted reward.
        qs = self.network.critic(obs, action)

        r_qv = qs - (batch["discount"] * next_v).expand(qs.shape[0],-1)

       



        r_mean1 = torch.mean(r_qv, dim=1, keepdim=True)
        r_var_single = ((r_qv - r_mean1) ** 2).sum() / (batch["reward"].shape[0]-1)/2
        r_var = (r_var_single).mean()

        rewards= batch["reward"].expand(qs.shape[0],-1)
        # r_qv=r_qv.expand(qs.shape[0],-1)


        # r1, r2 = torch.split(r_qv, 2, dim=1)
        # logits = r2.sum(dim=2)-r1.sum(dim=2)
        # re1, re2 = torch.split(rewards, 2, dim=1)
        # re1, re2 = re1.sum(dim=2), re2.sum(dim=2)
        # prob = 1.0 * (re1 < re2)
        # loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # ipl_loss = loss(logits, prob).mean()
        # reg1 = self.r_var * r_var  
        # reg1 = self.r_var * r_var  + 2 * r_qv.mean() * rewards.mean() + 2 * torch.sqrt(torch.square(r_qv).mean() * (torch.square(rewards.mean()) + r_var)) 
        reg1 = self.r_var * r_var  + 2 * r_qv.mean() * rewards.mean() - 2 * (r_qv * (r_qv - rewards.mean())).mean()

        reg2 = self.r_square *  (1/(2*rewards.mean()) * torch.square(r_qv) - r_qv).mean() + self.gamma * torch.square(r_qv-rewards.mean()).mean() 

        q_loss = torch.nn.functional.mse_loss(r_qv.mean(dim=1), rewards.mean(dim=1), reduction="none").mean() 
        q_loss += reg1 + reg2
        replay_reward_loss = torch.nn.functional.mse_loss(rewards, r_qv, reduction="none").mean()


        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(
            q_loss=q_loss.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            q=qs.mean().item(),
            adv=adv.mean().item(),
            # ipl_loss=ipl_loss.mean().item(),
            replay_reward_loss=replay_reward_loss.mean().item(),
            r_var=r_var.mean().item(),
            r_qv=r_qv.mean().item(),
            reward_rp_real=batch["reward"].mean().item(),
            reg1=reg1.mean().item(),
            reg2=reg2.mean().item(),



        )

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample() if sample else dist.loc
            elif torch.is_tensor(dist):
                action = dist
            else:
                raise ValueError("Invalid policy output")
            action = action.clamp(*self.action_range)
        return action

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
