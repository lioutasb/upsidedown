import gym
import os
import numpy as np
import random
import datetime
import torch
from torch import nn
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from experiment import rollout, ReplayBuffer, Trajectory, load_checkpoint, save_checkpoint
from sacred import Experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ex = Experiment()

class Behavior(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, state_shape, num_actions, return_scale, horizon_scale):
        super(Behavior, self).__init__()
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale

        self.fc_state = nn.Linear(state_shape, hidden_size)
        self.fc_dr = nn.Linear(1, hidden_size)
        self.fc_dh = nn.Linear(1, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state, dr, dh):
        # print(f"State shape: {state.shape}")
        # print(f"Dr shape: {dr.shape}")
        # print(f"Dh shape: {dh.shape}")
        output_state = self.fc_state(state)
        # print(f"State Output shape: {output_state.shape}")
        
        output_dr = torch.sigmoid(self.fc_dr(dr * self.return_scale))
        # print(f"Dr Output shape: {output_dr.shape}")
        output_dh = torch.sigmoid(self.fc_dh(dh * self.horizon_scale))
        # print(f"Dh Output shape: {output_dh.shape}")

        
        output = output_state * (output_dr + output_dh) # TODO: Is this a good way to combine these?
        
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return output

@ex.command
def train(_run, experiment_name, hidden_size, replay_size, last_few, lr, checkpoint_path):
    """
    Begin or resume training a policy.
    """
    run_id = _run._id or datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_dir = f'tensorboard/{run_id}_{experiment_name}'
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make('LunarLander-v2')
    
    loss_object = torch.nn.CrossEntropyLoss().to(device)
    
    model = Behavior(hidden_size=hidden_size, state_shape=env.observation_space.shape[0], num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)

    print("Trying to load:")
    print(checkpoint_path)
    c = load_checkpoint(checkpoint_path, model, optimizer, device, train=True)
    updates, steps, loss = c.updates, c.steps, c.loss

    last_eval_step = 0
    rewards = []
    done = False

    while not done:
        steps, updates, last_eval_step, done = do_iteration(env=env, model=model, optimizer=optimizer, 
            loss_object=loss_object, rb=rb, writer=writer, updates=updates, steps=steps,
            last_eval_step=last_eval_step, rewards=rewards)

    add_artifact()

@ex.capture
def do_iteration(env, model, optimizer, loss_object, rb, writer, updates, steps, last_eval_step, rewards):

    # Exloration
    steps = do_exploration(env, model, rb, writer, steps)
    
    # Updates    
    updates = do_updates(model, optimizer, loss_object, rb, writer, updates, steps)
        
    # Evaluation
    last_eval_step, done = do_eval(env=env, model=model, rb=rb, writer=writer, steps=steps, 
        rewards=rewards, last_eval_step=last_eval_step)

    return steps, updates, last_eval_step, done

@ex.capture
def do_eval(env, model, rb, writer, steps, rewards, last_eval_step, eval_episodes, 
    max_return, max_steps, solved_min_reward, solved_n_episodes, eval_every_n_steps):

    roll = rollout(eval_episodes, env=env, model=model, 
            sample_action=True, replay_buffer=rb, 
            device=device, evaluation=True,
            max_return=max_return)

    steps_exceeded = steps >= max_steps
    time_to_eval = ((steps - last_eval_step) >= eval_every_n_steps) or steps_exceeded or (last_eval_step == 0)

    if steps_exceeded:
        print(f"Steps {steps} exceeds max env steps {max_steps}.")

    if not time_to_eval:
        return last_eval_step, steps_exceeded

    last_eval_step = steps

    (dh, dr) = rb.eval_command()
    writer.add_scalar('Eval/dr', dr, steps)
    writer.add_scalar('Eval/dh', dh, steps)
    
    writer.add_scalar('Eval/reward', roll.mean_reward, steps) 
    writer.add_scalar('Eval/length', roll.mean_length, steps)
    
    print(f"Eval Episode Mean Reward: {roll.mean_reward}")      

    # Stopping criteria
    rewards.extend(roll.rewards)
    rewards = rewards[-solved_n_episodes:]
    eval_min_reward = np.min(rewards)

    solved = eval_min_reward >= solved_min_reward
    if solved:
        print(f"Task considered solved. Achieved {eval_min_reward} >= {solved_min_reward} over {solved_n_episodes} episodes.")
        
    return last_eval_step, solved
 

@ex.capture
def do_updates(model, optimizer, loss_object, rb, writer, updates, steps, checkpoint_path, 
    batch_size, n_updates_per_iter):
    loss_sum = 0
    loss_count = 0
   
    
    for _ in range(n_updates_per_iter):
        updates +=1

        sample = rb.sample(batch_size, device)    

        loss = train_step(sample, model, optimizer, loss_object)
        loss_sum += loss
        loss_count += 1            

    # Save updated model
    avg_loss = loss_sum/loss_count
    print(f'u: {updates}, s: {steps}, Loss: {avg_loss}')
    writer.add_scalar('Loss/avg_loss', avg_loss, steps)

    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, loss=avg_loss, updates=updates, steps=steps)
    return updates


@ex.capture
def do_exploration(env, model, rb, writer, steps, n_episodes_per_iter, epsilon, max_return):
    # Plot a sample dr/dh at this time
    example_cmd = rb.sample_command()

    writer.add_scalar('Exploration/dr', example_cmd.dr, steps)
    writer.add_scalar('Exploration/dh', example_cmd.dh, steps)

    # Exploration    
    roll = rollout(n_episodes_per_iter, env=env, model=model, 
        sample_action=True, replay_buffer=rb, device=device, 
        epsilon=epsilon, max_return=max_return)
    rb.add(roll.trajectories)

    steps += roll.length
    
    writer.add_scalar('Exploration/reward', roll.mean_reward, steps)
    writer.add_scalar('Exploration/length', roll.mean_length, steps)

    return steps

@ex.capture
def add_artifact(checkpoint_path):
    ex.add_artifact(checkpoint_path, name='checkpoint.pt')
           
def train_step(sample, model, optimizer, loss_object):
    optimizer.zero_grad()    
    predictions = model(state=sample.state, dr=sample.dr, dh=sample.dh)
    loss = loss_object(predictions, sample.action)
    
    loss.backward()
    optimizer.step()
    
    return loss


@ex.command
def play(checkpoint_path, epsilon, sample_action, hidden_size, play_episodes, dh, dr):
    """
    Play episodes using a trained policy. 
    """
    env = gym.make('LunarLander-v2')
    cmd = Command(dr=dr, dh=dh)

    loss_object = torch.nn.CrossEntropyLoss().to(device)
    model = Behavior(hidden_size=hidden_size,state_shape=env.observation_space.shape[0], num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    c = load_checkpoint(name=checkpoint_path, train=False, 
        model=model, optimizer=optimizer, device=device)

    for _ in range(play_episodes):
        roll = rollout(episodes=1, env=env, model=model, sample_action=sample_action, 
                              cmd=cmd, render=True, device=device)

        print(f"Episode Reward: {roll.mean_reward}")


@ex.config
def run_config():    
    train = True # Train or play?
    hidden_size = 32
    epsilon = 0.1
    return_scale = 0.01
    horizon_scale = 0.01

    # Train specific
    lr = 0.005
    batch_size = 1024
    solved_min_reward = 200 # Solved when min reward is at least this
    solved_n_episodes =  100 # for over this many episodes
    max_steps = 10**7
    replay_size = 100 # Maximum size of the replay buffer in episodes
    last_few = 50     
    n_episodes_per_iter = 10
    n_updates_per_iter = 50
    eval_episodes = 100
    eval_every_n_steps = 50_000
    max_return = 300

    experiment_name = f'lunarlander_hs{hidden_size}_mr{max_return}_b{batch_size}_rs{replay_size}_lf{last_few}_ne{n_episodes_per_iter}_nu{n_updates_per_iter}_e{epsilon}_lr{lr}'
    checkpoint_path = f'checkpoint_{experiment_name}.pt'


    # Play specific
    sample_action = True
    play_episodes = 5
    dh = 200
    dr = 400



@ex.automain
def main():
    """
    Default runs train() command
    """
    train()