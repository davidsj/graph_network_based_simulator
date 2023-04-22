import argparse
import os
import json
import datetime
import csv
import numpy as np
import torch
import tqdm

import gns

def pr(*x):
    datestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(datestr, *x)

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=str,
                    help='Directory containing the dataset. The --out_dir argument '
                         'used to generate the dataset should be passed here.')
parser.add_argument('connectivity_radius', type=float,
                    help='Maximum distance for particles to be considered neighbors.')
parser.add_argument('--n_previous_velocities', type=int, default=5,
                    help='Number of previous velocities to use as features'
                         'for each particle.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_schedule', type=str, default='triangular',
                    choices=['constant', 'triangular'],
                    help='Learning rate schedule. If triangular, the learning rate will be '
                         'increased linearly from 0 to args.lr over the first half of '
                         'training, then decreased linearly to 0 over the second half.')
parser.add_argument('--checkpoint_interval', type=int, default=2500,
                    help='Number of training steps between checkpoints.')
parser.add_argument('--log_record_interval', type=int, default=100,
                    help='Number of training steps between log records.')
parser.add_argument('--max_training_trajectories', type=int, default=None,
                    help='Maximum number of training trajectories to use. If None, use all.')
args = parser.parse_args()

# Validate arguments.
assert os.path.exists(args.dataset_dir)
assert args.n_previous_velocities >= 0
assert args.connectivity_radius >= 0.0
assert args.lr > 0.0
assert args.checkpoint_interval > 0
assert args.log_record_interval > 0
assert args.max_training_trajectories is None or args.max_training_trajectories >= 0

# Load metadata.
with open(os.path.join(args.dataset_dir, 'metadata.json'), 'r') as f:
    md = json.load(f)

# Create the output directory and record the arguments.
while True:
    datestr = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')
    outdir = os.path.join(args.dataset_dir, 'models', datestr)
    try:
        os.makedirs(outdir)
    except FileExistsError:
        # Try again with a different timestamp.
        continue
    break
with open(os.path.join(outdir, 'args.json'), 'w') as f:
    json.dump(vars(args), f)

# Load and shuffle the training data.
pr('Loading training trajectories...')
train_data = []
if args.max_training_trajectories is None:
    n_trajs = md['n_training']
else:
    n_trajs = min(md['n_training'], args.max_training_trajectories)
for i in tqdm.tqdm(range(n_trajs)):
    traj = gns.Trajectory.load(os.path.join(args.dataset_dir, 'training', f'{i}.npz'))
    train_data += traj.get_datapoints(args.n_previous_velocities)
np.random.shuffle(train_data)
print(f'Loaded {len(train_data)} training datapoints from {n_trajs} trajectories.')

# Initialize the model, optimizer, and learning rate scheduler.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = gns.GNS(md['n_materials'], md['dimensions'],
                args.n_previous_velocities, args.connectivity_radius,
                md['box_boundaries']).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.lr_schedule == 'triangular':
    sched = torch.optim.lr_scheduler.CyclicLR(opt, 0., args.lr, step_size_up=len(train_data)//2, cycle_momentum=False)
else:
    sched = None

# Train the model.
pr('Training...')
model.train()
training_loss_sum = 0.0
training_loss_n = 0

csv_path = os.path.join(outdir, 'log.csv')
with open(csv_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['step', 'training_loss', 'lr'])

for i, datapoint in enumerate(train_data):
    # Record training loss and checkpoint.
    if training_loss_n == args.log_record_interval:
        avg_training_loss = training_loss_sum / training_loss_n
        lr = opt.param_groups[0]['lr']
        pr(f'  Step {i}: avg training loss = {avg_training_loss:.3e} lr = {lr}')
        with open(csv_path, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([i, avg_training_loss, lr])
        training_loss_sum = 0.0
        training_loss_n = 0
    if i % args.checkpoint_interval == 0:
        torch.save(model.state_dict(), os.path.join(outdir, f'checkpoint-{i}.pt'))
        pr(f'  Step {i}: saved checkpoint.')

    # Train on the current datapoint.
    datapoint = datapoint.to(device)
    acc_hat = model(datapoint)
    acc = datapoint.accelerations
    loss = (acc_hat - acc).square().sum(axis=1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if sched is not None:
        sched.step()

    # Update the training loss.
    training_loss_sum += loss.item()
    training_loss_n += 1

# Record the final training loss and save the final checkpoint.
if training_loss_n > 0:
    avg_training_loss = training_loss_sum / training_loss_n
    lr = opt.param_groups[0]['lr']
    pr(f'  Step {len(train_data)}: final avg training loss = {avg_training_loss:.3e} lr = {lr}')
with open(csv_path, 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([len(train_data), avg_training_loss, lr])
torch.save(model.state_dict(), os.path.join(outdir, f'checkpoint-{len(train_data)}.pt'))
pr(f'  Step {len(train_data)}: saved final checkpoint.')
