import gym
import os
import numpy as np
import random
import torch
from torch import nn
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from experiment import rollout, ReplayBuffer, Trajectory, load_checkpoint, save_checkpoint
from sacred import Experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ex = Experiment()

class Behavior(nn.Module):
    def __init__(self, hidden_size, state_shape, cmd_shape, num_actions):
        super(Behavior, self).__init__()
        self.fc_state = nn.Linear(state_shape, hidden_size)
        self.fc_cmd = nn.Linear(cmd_shape, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        output_spate = self.fc_state(x[0])
        output_cmd = torch.sigmoid(self.fc_cmd(x[1]))
        
        output = output_spate * output_cmd
        
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return output

@ex.capture
def run_train(experiment_name, checkpoint_name, batch_size, max_steps, hidden_size, solved_mean_reward, solved_n_episodes, replay_size, last_few, 
    n_warmup_episodes, n_episodes_per_iter, n_updates_per_iter, start_epsilon, eval_episodes, max_return, lr):
    
    writer = SummaryWriter(comment=experiment_name)
    env = gym.make('LunarLander-v2')
    
    loss_object = torch.nn.CrossEntropyLoss().to(device)
    
    model = Behavior(hidden_size=hidden_size, state_shape=env.observation_space.shape[0], cmd_shape=2, num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)
    
    # Random rollout
    roll = rollout(episodes=n_warmup_episodes, env=env, render=False, max_return=max_return)
    rb.add(roll.trajectories)
    print(f"Mean Episode Reward: {roll.mean_reward}")

    # Keep track of steps used during random rollout!
    c = load_checkpoint(checkpoint_name, model, optimizer, device, train=True)
    updates, steps, loss = c.updates, c.steps, c.loss

    steps += roll.length
    
    save_checkpoint(checkpoint_name, model=model, optimizer=optimizer, loss=loss, updates=updates, steps=steps)

    # Plot initial values
    writer.add_scalar('Train/reward', roll.mean_reward, steps)   
    writer.add_scalar('Train/length', roll.mean_length, steps)

    loss_sum = 0
    loss_count = 0
    rewards = []

    while True:
        for _ in range(n_episodes_per_iter):
            updates +=1

            x, y = rb.sample(batch_size, device)    
            loss = train_step(x, y, model, optimizer, loss_object)
            loss_sum += loss
            loss_count += 1            
            writer.add_scalar('Loss/loss', loss, updates)

        # Save updated model
        avg_loss = loss_sum/loss_count
        print(f'u: {updates}, s: {steps}, Loss: {avg_loss}')

        save_checkpoint(checkpoint_name, model=model, optimizer=optimizer, loss=avg_loss, updates=updates, steps=steps)

        # Exploration    
        roll = rollout(n_episodes_per_iter, env=env, model=model, 
            sample_action=True, replay_buffer=rb, device=device, action_fn=action_fn, 
            epsilon=start_epsilon, max_return=max_return)
        rb.add(roll.trajectories)

        steps += roll.length
        
        (dh, dr) = rb.sample_command()
        writer.add_scalar('Train/dr', dr, steps)
        writer.add_scalar('Train/dh', dh, steps)

        writer.add_scalar('Train/reward', roll.mean_reward, steps)
        writer.add_scalar('Train/length', roll.mean_length, steps)
        
        # Eval
        roll = rollout(eval_episodes, env=env, model=model, 
                sample_action=True, replay_buffer=rb, 
                device=device, action_fn=action_fn, evaluation=True,
                max_return=max_return)

        (dh, dr) = rb.eval_command()
        writer.add_scalar('Eval/dr', dr, steps)
        writer.add_scalar('Eval/dh', dh, steps)
        
        writer.add_scalar('Eval/reward', roll.mean_reward, steps) 
        writer.add_scalar('Eval/length', roll.mean_length, steps)
        
        print(f"Eval Episode Mean Reward: {roll.mean_reward}")      

        # Stopping criteria
        rewards.extend(roll.rewards)
        rewards = rewards[-solved_n_episodes:]
        eval_mean_reward = np.mean(rewards)

        if eval_mean_reward >= solved_mean_reward:
            print("Task considered solved. Achieved {eval_mean_reward} >= {solved_mean_reward} over {solved_n_episodes} episodes.")
            break
        
        if steps >= max_steps:
            print(f"Steps {steps} exceeds max env steps {max_steps}. Stopping.")
            break  
           
def train_step(inputs, targets, model, optimizer, loss_object):
    optimizer.zero_grad()    
    predictions = model([inputs[:, :-2], inputs[:, -2:]])
    loss = loss_object(predictions, targets)
    
    loss.backward()
    optimizer.step()
    
    return loss

def action_fn(env, model, inputs, sample_action, epsilon):
    action_logits = model([inputs[:, :-2], inputs[:, -2:]])
    action_probs = torch.softmax(action_logits, axis=-1)

    if random.random() < epsilon: # Random action
        return env.action_space.sample()
    
    if sample_action:        
        m = torch.distributions.categorical.Categorical(logits=action_logits)             
        action = int(m.sample().squeeze().cpu().numpy())        
    else:
        action = int(np.argmax(action_probs.detach().squeeze().numpy()))
    return action

@ex.capture
def run_play(checkpoint_name, epsilon, sample_action, hidden_size, play_episodes, dh, dr):
    env = gym.make('LunarLander-v2')
    cmd = (dh, dr)

    loss_object = torch.nn.CrossEntropyLoss().to(device)
    model = Behavior(hidden_size=hidden_size,state_shape=env.observation_space.shape[0], cmd_shape=2, num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    c = load_checkpoint(name=checkpoint_name, train=False, 
        model=model, optimizer=optimizer, device=device)

    for _ in range(play_episodes):
        roll = rollout(episodes=1, env=env, model=model, sample_action=sample_action, 
                              cmd=cmd, render=True, device=device, action_fn=action_fn)

        print(f"Episode Reward: {roll.mean_reward}")


@ex.config
def run_config():    
    experiment_name = 'lunar_lander_v2'
    checkpoint_name = f'checkpoint_{experiment_name}.pt'
    train = True # Train or play?
    hidden_size = 32

    # Train specific
    lr = 0.005
    batch_size = 1024
    solved_mean_reward = 200 # Solved is 200 points
    solved_n_episodes =  100 # (Andriy: let's say over 100 episodes)
    max_steps = 10**7
    replay_size = 100 # Maximum size of the replay buffer in episodes
    last_few = 50     
    n_warmup_episodes = 30
    n_episodes_per_iter = 10
    n_updates_per_iter = 50
    start_epsilon = 0.1 # Probability of taking a random action during training
    eval_episodes = 10
    max_return = 300

    # Play specific
    epsilon = 0.0
    sample_action = True
    play_episodes = 5
    dh = 200
    dr = 400



@ex.automain
@ex.capture
def main(train):
    if train:
        run_train()
    else:
        run_play()