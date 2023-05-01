import os
import numpy as np
import argparse
import json
import urllib.request

import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gns

# Importing tensorflow (2.12.0) before torch (2.0.0) leads to a freeze,
# so we save this import for last. Disabling info logging and use of GPU,
# since we have no need for either.
os.environ['OMPI_MCA_btl_openib_warn_no_device_params_found'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str)
parser.add_argument('--reference_datasets_dir', type=str, default='reference_datasets')
args = parser.parse_args()

# Make dataset directory (will fail if already exists).
out_dir = os.path.join(args.reference_datasets_dir, args.dataset_name)
os.makedirs(out_dir)
download_dir = os.path.join(out_dir, 'download')
os.makedirs(download_dir)


base_url = f'https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{args.dataset_name}/'

# Download metadata and make sure it doesn't contain any information
# that we don't understand.
metadata_url = base_url + 'metadata.json'
md_orig = json.loads(urllib.request.urlopen(metadata_url).read())
expected_keys = {
    'bounds',
    'sequence_length',
    'default_connectivity_radius',
    'dim',
    'dt',
    'vel_mean',
    'vel_std',
    'acc_mean',
    'acc_std'
}
assert set(md_orig.keys()) == expected_keys
with open(os.path.join(download_dir, 'metadata.json'), 'w') as f:
    json.dump(md_orig, f)

# Convert metadata to our format.
md = {
    'out_dir': out_dir,
    'traj_len': md_orig['sequence_length'] + 1,
    'dt_ms': md_orig['dt'] * 1000,
    'name': args.dataset_name,
    'dimensions': md_orig['dim'],
    'n_materials': 0, # This will be updated as we process the trajectories.
    'box_boundaries': md_orig['bounds'],
    'default_connectivity_radius': md_orig['default_connectivity_radius'],
    'training_stats': {
        'vel': {
            'mean': md_orig['vel_mean'],
            'std': md_orig['vel_std']
        },
        'acc': {
            'mean': md_orig['acc_mean'],
            'std': md_orig['acc_std']
        }
    }
}

# Download trajectories and convert them to our format.
for split in ['training', 'validation', 'test']:
    short_split = split[:5]
    os.makedirs(os.path.join(out_dir, split))

    # Download the tfrecord.
    tfrecord_url = base_url + f'{short_split}.tfrecord'
    tfrecord_path = os.path.join(download_dir, f'{short_split}.tfrecord')
    print(f'Downloading {tfrecord_url} to {tfrecord_path}...')
    urllib.request.urlretrieve(tfrecord_url, tfrecord_path)

    # Decode the trajectories.
    print(f'Decoding {tfrecord_path}...')
    ds = tf.data.TFRecordDataset(tfrecord_path)
    n = 0
    with open(os.path.join(out_dir, split, 'lengths'), 'w') as lf:
        for i, rec in enumerate(tqdm.tqdm(ds)):
            context_features = {
                'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                'particle_type': tf.io.VarLenFeature(tf.string)
            }
            feature_description = {
                'position': tf.io.VarLenFeature(tf.string),
                'step_context': tf.io.VarLenFeature(tf.string)
            }
            context, parsed_features = tf.io.parse_single_sequence_example(
                rec, context_features=context_features, sequence_features=feature_description)
            
            mat = tf.io.decode_raw(context['particle_type'].values, tf.int64).numpy()
            md['n_materials'] = max(md['n_materials'], (mat.max() + 1).item())
            pos = tf.io.decode_raw(parsed_features['position'].values, tf.float32).numpy()
            step_context = tf.io.decode_raw(parsed_features['step_context'].values, tf.float32).numpy()

            # We don't know how to handle nontrivial step contexts yet.
            assert step_context.shape[1] == 1
            assert np.all(np.isnan(step_context) | (step_context == 0))

            # Save trajectory and length.
            traj = gns.Trajectory(pos.reshape((pos.shape[0], -1, md['dimensions'])), mat.reshape((-1)))
            traj.save(os.path.join(out_dir, split, f'{i}.npz'))
            lf.write(f'{pos.shape[0]}\n')

            n += 1
    md[f'n_{split}'] = n

# Save metadata.
with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(md, f)
