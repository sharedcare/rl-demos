from ast import arg
import math
import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.init as init

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from itertools import count
from models import DQN
from memory import ReplayMemory
from env import make_env

writer = SummaryWriter()


class Agent(object):
    def __init__(self, env, device, batch_size, mem_size, lr, gamma, epsilon, epsilon_decay, epsilon_min, target_update, save_path):
        # params
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.memory_size = mem_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = epsilon
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.model_save_path = save_path
        self.print_every = 50
        self.eval_every = 1000
        self.steps = 0
        
        # DQN init
        self.policy_net = DQN(obs_dim=env.observation_space.shape, action_dim=env.action_space.n)
        self.target_net = DQN(obs_dim=env.observation_space.shape,
                              action_dim=env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(mem_size)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        self.target_net.load_state_dict(torch.load(path))
        self.target_net.eval()

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
            math.exp(-1. * self.steps / self.epsilon_decay)
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def update_policy(self, batch):
        states, actions, next_states, rewards, dones = batch
        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.cat([s for s in next_states
                                 if s is not None]).to(self.device)
        dones = torch.tensor(
            tuple(map(lambda d: not d, dones)),
            device=self.device, dtype=torch.bool)

        q_values = self.policy_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        next_q_values[dones] = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values

        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def train(self, n_episodes, render=False):
        best_score = -np.inf
        scores = []
        losses = []
        for episode in range(n_episodes):
            obs = self.env.reset()
            state = self.get_state(obs)
            score = 0.0
            loss = 0
            step = 0
            done = False
            while not done:
                self.update_epsilon()
                action = self.choose_action(state)

                if render:
                    env.render()

                obs, reward, done, info = env.step(action)

                score += reward
                step += 1
                self.steps += 1

                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None

                reward = torch.tensor([reward], device=device)
                done = torch.tensor([done], device=device)

                self.memory.push(state, action, next_state,
                            reward.to(device), done.to(device))
                state = next_state

                if self.steps > self.batch_size:
                    batch = self.memory.sample(self.batch_size)
                    loss = self.update_policy(batch)
                    losses.append(loss)

                    if self.steps % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

            scores.append(score)

            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
            else:
                avg_score = np.mean(scores)

            if episode % self.print_every == 0:
                print('Total steps: {} \t Episode: {}/{} \t Score: {}, Eps: {}'.format(
                    self.steps, episode, step, score, self.epsilon), end='\r', flush=True)

            if episode % self.eval_every == 0:
                eval_scores = self.test(10)
                writer.add_scalar("Score/test", np.mean(eval_scores), episode)

            if avg_score > best_score:
                best_score = avg_score
                save_path = os.path.join(self.model_save_path,
                                    "dqn_" + str(episode) + "_weights.pth")
                self.save_model(save_path)

            writer.add_scalar("Loss/train", loss, episode)
            writer.add_scalar("Score/train", avg_score, episode)
            writer.add_scalar("Eps/train", self.epsilon, episode)

        env.close()
        return

    def test(self, n_episodes, render=False):
        scores = []
        steps = []
        for episode in range(n_episodes):
            obs = self.env.reset()
            state = self.get_state(obs)
            score = 0.0
            done = False
            step = 0
            while not done:
                action = self.policy_net(state.to(self.device)).max(1)[
                    1].view(1, 1)
                
                # if render:
                #     self.env.render()

                obs_, reward, done, info = self.env.step(action)

                score += reward
                step += 1
                scores.append(score)

                if not done:
                    next_state = self.get_state(obs_)
                else:
                    next_state = None
                
                state = next_state

            scores.append(score)
            steps.append(step)

        print('Test: Average steps: {}, Average score: {}'.format(
            np.mean(steps), np.mean(scores)))
        return scores

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BreakoutNoFrameskip-v4", help="gym environment")
    parser.add_argument("--device", default="cpu", help="device")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--mem_size", default=100000, type=int, help="memory size")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--epsilon", default=1.0, type=float, help="initial epsilon")
    parser.add_argument("--epsilon_decay", default=1000000,
                        type=float, help="epsilon decay rate")
    parser.add_argument("--epsilon_min", default=0.01, type=float, help="min epsilon")
    parser.add_argument("--target_update", default=1000, type=int, help="target update frequency")
    parser.add_argument("--n_episodes", default=20000, type=int, help="number of episodes")
    parser.add_argument("--n_test_episodes", default=10, type=int, help="number of test episodes")
    parser.add_argument("--model_save_path", default="./saved_models", help="model save path")
    parser.add_argument("--render", default=False, action="store_true", help="render")
    parser.add_argument("--test", default=False,
                        action="store_true", help="test")
    parser.add_argument("--load_model", default=False, action="store_false", help="whether to load trained model")
    args = parser.parse_args()

    if args.render: # render mode
        env = gym.make(args.env, render_mode='human')
    else:
        env = gym.make(args.env)
    env = make_env(env)
    device = torch.device(args.device)
    
    agent = Agent(
        env, 
        device, 
        args.batch_size, 
        args.mem_size, 
        args.lr, 
        args.gamma, 
        args.epsilon, 
        args.epsilon_decay, 
        args.epsilon_min, 
        args.target_update, 
        args.model_save_path
        )
    if args.test:
        agent.load_model(args.model_save_path + "/dqn_best.pth")
        agent.test(args.n_test_episodes, args.render)
    else:
        if args.load_model:
            agent.load_model(args.model_save_path + "/dqn_best.pth")
        agent.train(args.n_episodes, args.render)
        writer.flush()
        writer.close()
        agent.save_model(args.model_save_path + "/dqn_best.pth")
    env.close()
    print("Closed environment")