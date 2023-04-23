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
parser.add_argument('--checkpoint_interval', type=int, default=5000,
                    help='Number of training steps between checkpoints.')
parser.add_argument('--validation_interval', type=int, default=None,
                    help='Number of training steps between calculating validation loss. If'
                         'None, use the same value as --checkpoint_interval.')
parser.add_argument('--log_record_interval', type=int, default=100,
                    help='Number of training steps between training log records.')
parser.add_argument('--max_training_trajectories', type=int, default=None,
                    help='Maximum number of training trajectories to use. If None, use all.')
parser.add_argument('--max_validation_trajectories', type=int, default=5,
                    help='Maximum number of validation trajectories to use. If None, use all.')
args = parser.parse_args()

# Validate arguments.
assert os.path.exists(args.dataset_dir)
assert args.n_previous_velocities >= 0
assert args.connectivity_radius >= 0.0
assert args.lr > 0.0
assert args.checkpoint_interval > 0
if args.validation_interval is None:
    args.validation_interval = args.checkpoint_interval
assert args.validation_interval > 0
assert args.log_record_interval > 0
assert args.max_training_trajectories is None or args.max_training_trajectories >= 0
assert args.max_validation_trajectories is None or args.max_validation_trajectories > 0


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
pr(f'Output directory: {outdir}')
with open(os.path.join(outdir, 'args.json'), 'w') as f:
    json.dump(vars(args), f)

# Load the training and validation data.
def load_split(split_name, md, max_trajectories=None, shuffle=False, device=None):
    pr(f'Loading {split_name} trajectories...')
    split_data = []
    if max_trajectories is None:
        n_trajs = md[f'n_{split_name}']
    else:
        n_trajs = min(md[f'n_{split_name}'], max_trajectories)
    for i in tqdm.tqdm(range(n_trajs)):
        traj = gns.Trajectory.load(os.path.join(args.dataset_dir, split_name, f'{i}.npz'))
        split_data += traj.get_datapoints(args.n_previous_velocities)
    if shuffle:
        np.random.shuffle(split_data)
    if device is not None:
        split_data = [datapoint.to(device) for datapoint in split_data]
    print(f'Loaded {len(split_data)} {split_name} datapoints from {n_trajs} trajectories.')
    return split_data

train_data = load_split('training', md, args.max_training_trajectories, shuffle=True)
val_data = load_split('validation', md, args.max_validation_trajectories, shuffle=False, device='cuda')

# Define loss functions.
def loss_fn(acc, acc_hat):
    return (acc_hat - acc).square().sum(axis=1).mean()

def calc_val_loss(model, val_data):
    pr('Calculating validation loss...')
    prev_model_training_state = model.training
    model.eval()
    val_loss_sum = 0.0
    val_loss_n = 0
    with torch.no_grad():
        for datapoint in tqdm.tqdm(val_data):
            loss = loss_fn(datapoint.accelerations, model(datapoint))
            val_loss_sum += loss.item()
            val_loss_n += 1
    model.train(prev_model_training_state)
    return val_loss_sum / val_loss_n

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
validation_csv_path = os.path.join(outdir, 'validation_log.csv')
with open(csv_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['step', 'training_loss', 'lr'])
with open(validation_csv_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['step', 'validation_loss'])

for i, datapoint in enumerate(train_data):
    # Record training loss.
    if training_loss_n == args.log_record_interval:
        avg_training_loss = training_loss_sum / training_loss_n
        lr = opt.param_groups[0]['lr']
        pr(f'  Step {i}: avg training loss = {avg_training_loss:.3e} lr = {lr}')
        with open(csv_path, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([i, avg_training_loss, lr])
        training_loss_sum = 0.0
        training_loss_n = 0
    # Record validation loss.
    if i % args.validation_interval == 0:
        val_loss = calc_val_loss(model, val_data)
        pr(f'  Step {i}: validation loss = {val_loss:.3e}')
        with open(validation_csv_path, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([i, val_loss])
    # Save a checkpoint.
    if i % args.checkpoint_interval == 0:
        torch.save(model.state_dict(), os.path.join(outdir, f'checkpoint-{i}.pt'))
        pr(f'  Step {i}: saved checkpoint.')

    # Train on the current datapoint.
    datapoint = datapoint.to(device)
    loss = loss_fn(datapoint.accelerations, model(datapoint))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if sched is not None:
        sched.step()

    # Update the running training loss.
    training_loss_sum += loss.item()
    training_loss_n += 1

# Record the final training loss.
if training_loss_n > 0:
    avg_training_loss = training_loss_sum / training_loss_n
    lr = opt.param_groups[0]['lr']
    pr(f'  Step {len(train_data)}: final avg training loss = {avg_training_loss:.3e} lr = {lr}')
with open(csv_path, 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([len(train_data), avg_training_loss, lr])
# Record the final validation loss.
val_loss = calc_val_loss(model, val_data)
pr(f'  Step {len(train_data)}: final validation loss = {val_loss:.3e}')
with open(validation_csv_path, 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([len(train_data), val_loss])
# Save the final checkpoint.
torch.save(model.state_dict(), os.path.join(outdir, f'checkpoint-{len(train_data)}.pt'))
pr(f'  Step {len(train_data)}: saved final checkpoint.')
