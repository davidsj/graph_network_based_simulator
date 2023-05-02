import argparse
import os
import json
import datetime
import csv
import numpy as np
import torch
import tqdm

import gns

def nowstr():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def pr(*x):
    print(nowstr(), *x)

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=str,
                    help='Directory containing the dataset. The --out_dir argument '
                         'used to generate the dataset should be passed here.')
parser.add_argument('--connectivity_radius', type=float,
                    help='Maximum distance for particles to be considered neighbors. '
                         'If not specified, the default value for the dataset will be '
                         'used.')
parser.add_argument('--n_previous_velocities', type=int, default=5,
                    help='Number of previous velocities to use as features '
                         'for each particle.')
parser.add_argument('--cum_velocity_noise_std', type=float, default=0.0003,
                    help='Standard deviation of cumulative noise in each dimension '
                         'of the velocity of each particle. Added during training '
                         'to help the model be robust to accumulated rollout error.')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'])
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_schedule', type=str, default='exponential_plus_constant',
                    choices=['constant', 'triangular', 'exponential_plus_constant'],
                    help='Learning rate schedule. If triangular, the learning rate will be '
                         'increased linearly from 0 to --lr over the first half of '
                         'training, then decreased linearly to 0 over the second half. '
                         'If exponential_plus_constant, the learning rate will decay from '
                         '--lr to --lr_exponential_min, with the delta above '
                         '--lr_exponential_min decreasing continuously by a factor of '
                         '--lr_exponential_decay_factor every --lr_exponential_decay_interval steps.')
parser.add_argument('--lr_exponential_min', type=float, default=1e-6)
parser.add_argument('--lr_exponential_decay_factor', type=float, default=0.1)
parser.add_argument('--lr_exponential_decay_interval', type=int, default=5000000)
parser.add_argument('--checkpoint_interval', type=int, default=10000,
                    help='Number of training steps between checkpoints.')
parser.add_argument('--validation_interval', type=int, default=10000,
                    help='Number of training steps between calculating validation loss.')
parser.add_argument('--log_record_interval', type=int, default=100,
                    help='Number of training steps between training log records.')
parser.add_argument('--training_batch_size', type=int, default=2)
parser.add_argument('--validation_batch_size', type=int, default=2)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--max_training_trajectories', type=int, default=None,
                    help='Maximum number of training trajectories to use. If None, use all.')
parser.add_argument('--max_validation_trajectories', type=int, default=5,
                    help='Maximum number of validation trajectories to use. If None, use all.')
parser.add_argument('--run_info', type=str, default=None,
                    help='Additional info string to include in metadata.')
args = parser.parse_args()

# Validate arguments.
assert os.path.exists(args.dataset_dir)
assert args.n_previous_velocities >= 0
assert args.cum_velocity_noise_std >= 0.0
if args.connectivity_radius is not None:
    assert args.connectivity_radius >= 0.0
assert args.lr > 0.0
assert args.lr_exponential_min >= 0.0
assert args.lr_exponential_decay_factor > 0.0
assert args.lr_exponential_decay_interval > 0
assert args.checkpoint_interval > 0
assert args.validation_interval > 0
assert args.log_record_interval > 0
assert args.training_batch_size > 0
assert args.validation_batch_size > 0
assert args.n_epochs >= 0
assert args.max_training_trajectories is None or args.max_training_trajectories >= 0
assert args.max_validation_trajectories is None or args.max_validation_trajectories > 0

# Load metadata.
with open(os.path.join(args.dataset_dir, 'metadata.json'), 'r') as f:
    md = json.load(f)
if args.connectivity_radius is None:
    args.connectivity_radius = md['default_connectivity_radius']

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
csv_path = os.path.join(outdir, 'log.csv')
validation_csv_path = os.path.join(outdir, 'validation_log.csv')
with open(csv_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['global_step', 'epoch', 'step', 'timestamp', 'training_loss', 'lr'])
with open(validation_csv_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['global_step', 'epoch', 'step', 'timestamp', 'validation_loss'])

# Load the training and validation data.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_data = gns.TorchDataset(args.dataset_dir, 'training',
                              n_previous_velocities=args.n_previous_velocities,
                              cum_vel_noise_std=args.cum_velocity_noise_std,
                              end_traj_idx=args.max_training_trajectories,
                              batch_size=args.training_batch_size,
                              device=device, shuffle=True)
val_data = gns.TorchDataset(args.dataset_dir, 'validation',
                            n_previous_velocities=args.n_previous_velocities,
                            end_traj_idx=args.max_validation_trajectories,
                            batch_size=args.validation_batch_size,
                            device=device)

# Define loss functions.
def loss_fn(acc, acc_hat):
    return (acc_hat - acc).square().sum(axis=-1).mean()

def calc_val_loss(model, val_data):
    # Batch (hence buffer) sizes might be different for training and
    # validation, so we empty the cache before and after to free up
    # memory for other processes.
    torch.cuda.empty_cache()
    prev_model_training_state = model.training
    model.eval()

    pr('Calculating validation loss...')
    val_loss_sum = 0.0
    val_loss_n = 0
    with torch.no_grad():
        for dps in tqdm.tqdm(val_data):
            batch_loss, batch_n = model.get_loss(dps)
            val_loss_sum += batch_loss.item()
            val_loss_n += batch_n

    model.train(prev_model_training_state)
    torch.cuda.empty_cache()
    return val_loss_sum / val_loss_n

# Initialize the model, optimizer, and learning rate scheduler.
model = gns.GNS(md['n_materials'], md['dimensions'],
                args.n_previous_velocities, args.connectivity_radius,
                md['training_stats'], args.cum_velocity_noise_std,
                box_boundaries=md['box_boundaries']).to(device)
if args.optimizer == 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'AdamW':
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
if args.lr_schedule == 'triangular':
    sched = torch.optim.lr_scheduler.CyclicLR(opt, 0., args.lr, cycle_momentum=False,
                                              step_size_up=np.ceil(args.n_epochs*len(train_data)/2))
elif args.lr_schedule == 'exponential_plus_constant':
    sched = gns.ExponentialPlusConstantLR(opt, args.lr, args.lr_exponential_min,
                                          args.lr_exponential_decay_factor,
                                          args.lr_exponential_decay_interval)
else:
    sched = None

# Train the model.
pr('Training...')
model.train()

global_step = 0
for epoch in range(args.n_epochs):
    training_loss_sum = 0.0
    training_loss_n = 0
    for i, dps in enumerate(train_data):
        # Record training loss.
        if training_loss_n == args.log_record_interval:
            avg_training_loss = training_loss_sum / training_loss_n
            lr = opt.param_groups[0]['lr']
            pr(f'  Epoch {epoch} Step {i}: avg training loss = {avg_training_loss:.3e} lr = {lr}')
            with open(csv_path, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([global_step, epoch, i, nowstr(), avg_training_loss, lr])
            training_loss_sum = 0.0
            training_loss_n = 0
        # Record validation loss.
        if i % args.validation_interval == 0:
            val_loss = calc_val_loss(model, val_data)
            pr(f'  Epoch {epoch} Step {i}: validation loss = {val_loss:.3e}')
            with open(validation_csv_path, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([global_step, epoch, i, nowstr(), val_loss])
        # Save a checkpoint.
        if i % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(outdir, f'checkpoint-{epoch}.{i}.pt'))
            pr(f'  Epoch {epoch} Step {i}: saved checkpoint.')

        # Train on the current batch.
        loss, _ = model.get_loss(dps, mean=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        # Update the running training loss.
        training_loss_sum += loss.item()
        training_loss_n += 1

        global_step += 1

    # Record the final training loss for this epoch.
    if training_loss_n > 0:
        avg_training_loss = training_loss_sum / training_loss_n
        lr = opt.param_groups[0]['lr']
        pr(f'  Epoch {epoch} Step {len(train_data)}: final avg training loss = {avg_training_loss:.3e} lr = {lr}')
    with open(csv_path, 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([global_step, epoch, len(train_data), nowstr(), avg_training_loss, lr])

# Record the final validation loss.
val_loss = calc_val_loss(model, val_data)
pr(f'  Epoch {epoch} Step {len(train_data)}: final validation loss = {val_loss:.3e}')
with open(validation_csv_path, 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([global_step, epoch, len(train_data), nowstr(), val_loss])
# Save the final checkpoint.
torch.save(model.state_dict(), os.path.join(outdir, f'checkpoint-{epoch}.{len(train_data)}.pt'))
pr(f'  Epoch {epoch} Step {len(train_data)}: saved final checkpoint.')
