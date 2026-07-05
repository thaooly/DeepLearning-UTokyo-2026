import os, subprocess, sys

USE_COLAB = False  # set to True when running in Colab

if USE_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = '/content/drive/MyDrive/GardenBird'
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

SPEC_DIR       = os.path.join(PROJECT_ROOT, 'Dataset', 'mygardenbird_spectrogram_pt')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR    = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

print(f'Project root  : {PROJECT_ROOT}')
print(f'Spec dataset  : {SPEC_DIR}')
print(f'Checkpoints   : {CHECKPOINT_DIR}')
print(f'Results       : {RESULTS_DIR}')

if USE_COLAB:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        'torchaudio', 'soundfile', 'scipy', 'torchinfo',
        'scikit-learn', 'tqdm', 'requests', 'pandas',
    ])

spectrogram_script = os.path.join(PROJECT_ROOT, 'Dataset', 'make_garden_spectrogram_pt_dataset.py')
cache_dir         = os.path.join(PROJECT_ROOT, 'Dataset', 'mygardenbird_download_cache')

if os.path.exists(os.path.join(SPEC_DIR, 'train_data_batches.pt')):
    print('Spectrogram PT batches already exist - skipping.')
else:
    print('Downloading from Zenodo and building mel-spectrogram PT batches...')
    subprocess.check_call([
        sys.executable, spectrogram_script,
        '--out-dir', SPEC_DIR,
        '--cache-dir', cache_dir,
        '--n-per-class', '575',
        '--batch-size', '32',
        '--seed', '42',
        '--force',
    ])

print('Dataset ready.')

import json, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import torch

from ClassesML.ResNetImprovedV2 import ResNetImprovedV2
from ClassesML.ConvAutoEncoder import ConvAutoEncoder
from ClassesML.POCS import pocs_postprocess
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifierV2 import TrainerClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

print('Loading dataset...')
x_train_cpu = torch.load(os.path.join(SPEC_DIR, 'train_data_batches.pt'))
y_train_cpu = torch.load(os.path.join(SPEC_DIR, 'train_label_batches.pt'))
x_valid_cpu = torch.load(os.path.join(SPEC_DIR, 'val_data_batches.pt'))
y_valid_cpu = torch.load(os.path.join(SPEC_DIR, 'val_label_batches.pt'))

classes   = open(os.path.join(SPEC_DIR, 'classes.txt')).read().strip().split('\n')
n_classes = len(classes)

def rebatch(x_batches, y_batches, size=32):
    x_all = torch.cat(x_batches)
    y_all = torch.cat(y_batches)
    return (
        [x_all[i:i+size] for i in range(0, len(x_all), size)],
        [y_all[i:i+size] for i in range(0, len(y_all), size)]
    )

x_train_cpu, y_train_cpu = rebatch(x_train_cpu, y_train_cpu)
x_valid_cpu, y_valid_cpu = rebatch(x_valid_cpu, y_valid_cpu)

input_dim = tuple(x_train_cpu[0].shape[1:])

if torch.cuda.is_available():
    print('Moving data to GPU...')
    x_train_gpu = [x.to(device) for x in x_train_cpu]
    y_train_gpu = [y.to(device) for y in y_train_cpu]
    x_valid_gpu = [x.to(device) for x in x_valid_cpu]
    y_valid_gpu = [y.to(device) for y in y_valid_cpu]
    print(f'GPU memory used : {torch.cuda.memory_allocated()/1e9:.2f} Go')
else:
    print('No GPU detected, data will stay on CPU.')
    x_train_gpu = x_train_cpu
    y_train_gpu = y_train_cpu
    x_valid_gpu = x_valid_cpu
    y_valid_gpu = y_valid_cpu

print(f'Train batches : {len(x_train_gpu)}  |  Val batches : {len(x_valid_gpu)}')
print(f'Input shape   : {input_dim}  |  Classes : {n_classes}')


def build_run_label(hp):
    """Build a readable label from active hyperparameter flags."""
    parts = []
    if hp.get('pool_every_stage'):  parts.append('Pool')
    if hp.get('use_se_block'):      parts.append('SE')
    if hp.get('use_tf_attention'):  parts.append('TFA')
    if hp.get('use_autoencoder'):   parts.append('AE')
    if hp.get('use_pocs'):          parts.append('POCS')
    if hp.get('use_spec_augment'):
        fm  = hp.get('spec_augment_freq_mask', 15)
        tm  = hp.get('spec_augment_time_mask', 30)
        nfm = hp.get('spec_augment_n_freq_masks', 2)
        ntm = hp.get('spec_augment_n_time_masks', 2)
        parts.append(f'SpecAug(F={fm}x{nfm} T={tm}x{ntm})')
    if hp.get('use_mixup'):
        parts.append(f'Mixup(a={hp.get("mixup_alpha", 0.4)})')
    return ' + '.join(parts) if parts else 'Baseline'


def find_existing_run(hp):
    """Return metadata of an already completed identical run, or None."""
    flags = [
        'pool_every_stage', 'use_se_block', 'use_tf_attention',
        'use_autoencoder', 'use_pocs',
        'use_spec_augment', 'spec_augment_freq_mask', 'spec_augment_time_mask',
        'spec_augment_n_freq_masks', 'spec_augment_n_time_masks',
        'use_mixup', 'mixup_alpha',
    ]
    for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'metrics_*.json'))):
        with open(path) as f:
            log = json.load(f)
        saved = log.get('hyperparameter', {})
        if all(saved.get(k) == hp.get(k) for k in flags):
            ts = path.split('metrics_')[-1].replace('.json', '')
            return {
                'timestamp'   : ts,
                'best_val_acc': log.get('best_val_acc', max(log['valid_accuracy_list'])),
                'run_label'   : log.get('run_label', build_run_label(hp)),
            }
    return None


def run_experiment(hp):
    """
    Train ResNetImprovedV2 with the given hyperparameters.
    Saves weights (.pt), metrics (.json), and learning curve (.png).
    Skips if an identical run already exists.
    """
    hp = hp.copy()
    hp['input_dim']  = input_dim
    hp['output_dim'] = n_classes

    run_label = build_run_label(hp)

    existing = find_existing_run(hp)
    if existing is not None:
        print(f'[SKIP] {existing["run_label"]}  ({existing["timestamp"]})  {existing["best_val_acc"]:.1f}%')
        return existing

    print(f'\n{"="*60}')
    print(f'Run : {run_label}')
    print(f'{"="*60}')

    # Fix seeds for reproducibility
    seed = hp.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x_train = x_train_gpu
    y_train = y_train_gpu
    x_valid = x_valid_gpu
    y_valid = y_valid_gpu

    # Optional ConvAutoEncoder pre-training
    ae_model = None
    if hp.get('use_autoencoder'):
        print('[AE] Pre-training...')
        ae_model  = ConvAutoEncoder(hp).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=hp.get('ae_learning_rate', 1e-3))
        for epoch in range(hp.get('ae_max_epoch', 20)):
            ae_model.train()
            total_loss = 0.0
            for x in x_train:
                x_hat = ae_model(x)
                loss  = criterion(x_hat, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch {epoch+1}  Loss: {total_loss/len(x_train):.4f}')
        ae_model.eval()

    # Optional AE / POCS preprocessing
    if hp.get('use_autoencoder') or hp.get('use_pocs'):
        print('[Preprocessing] Applying AE / POCS...')
        def process(batches):
            out = []
            with torch.no_grad():
                for x in batches:
                    if hp.get('use_autoencoder') and ae_model is not None:
                        x = ae_model(x)
                    if hp.get('use_pocs'):
                        x = pocs_postprocess(x, hp)
                    out.append(x)
            return out
        x_train = process(x_train)
        x_valid = process(x_valid)

    model = ResNetImprovedV2(hp).to(device)
    scope = ScopeClassifier(model, hp)

    trainer = TrainerClassifier(hyperparameter=hp)
    trainer.set_model(model=model, device=device)
    trainer.set_scope(scope=scope)
    trainer.set_data(x_train=x_train, y_train=y_train,
                     x_valid=x_valid, y_valid=y_valid)

    train_acc_list, valid_acc_list = trainer.run()

    best_val = max(valid_acc_list)
    best_ep  = valid_acc_list.index(best_val) + 1

    # Save weights and metrics
    timestamp       = datetime.now().strftime('%Y%m%d_%H%M%S')
    classifier_path = os.path.join(CHECKPOINT_DIR, f'resnet_{timestamp}.pt')
    ae_path         = os.path.join(CHECKPOINT_DIR, f'autoencoder_{timestamp}.pt')
    metrics_path    = os.path.join(CHECKPOINT_DIR, f'metrics_{timestamp}.json')

    torch.save(model.state_dict(), classifier_path)
    if ae_model is not None:
        torch.save(ae_model.state_dict(), ae_path)

    serialisable_hp = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}
    logs = {
        'hyperparameter'     : serialisable_hp,
        'run_label'          : run_label,
        'train_accuracy_list': train_acc_list,
        'valid_accuracy_list': valid_acc_list,
        'best_val_acc'       : best_val,
        'best_epoch'         : best_ep,
        'classes'            : classes,
        'ae_weights_path'    : ae_path if ae_model is not None else None,
    }
    with open(metrics_path, 'w') as f:
        json.dump(logs, f, indent=4)

    # Save learning curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_acc_list, 'b-', linewidth=2, label='Train Accuracy')
    ax.plot(valid_acc_list, 'r-', linewidth=2, label='Validation Accuracy')
    ax.axvline(x=best_ep - 1, color='green', linestyle='--', alpha=0.7,
               label=f'Best val: {best_val:.1f}% @ epoch {best_ep}')
    ax.set_title(f'{run_label}\n{timestamp}', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    curve_path = os.path.join(RESULTS_DIR, f'learning_curves_{timestamp}.png')
    fig.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Best val : {best_val:.1f}% @ epoch {best_ep}')
    print(f'Saved    : {classifier_path}')

    return {'timestamp': timestamp, 'best_val_acc': best_val, 'run_label': run_label}


print('Setup complete.')

hp_single = dict(

    # Architecture flags
    pool_every_stage          = True,
    pool_output_size          = None,
    use_se_block              = True,
    use_tf_attention          = True,

    # Preprocessing
    use_autoencoder           = False,
    use_pocs                  = False,

    # Data augmentation
    use_spec_augment          = True,
    spec_augment_freq_mask    = 25,
    spec_augment_time_mask    = 50,
    spec_augment_n_freq_masks = 3,
    spec_augment_n_time_masks = 3,
    use_mixup                 = True,
    mixup_alpha               = 0.4,

    # Training
    hidden_layers_size        = [128, 64],
    activation                = 'relu',
    kernel_size               = (5, 5),
    filters                   = [8, 16, 32, 32, 32],
    batch_normalization       = True,
    dropout_rate              = 0.3,
    learning_rate             = 1e-4,
    max_epoch                 = 50,
    seed                      = 42,

    # AE (used only if use_autoencoder=True)
    ae_base_channels          = 16,
    ae_learning_rate          = 1e-3,
    ae_max_epoch              = 20,

    # POCS (used only if use_pocs=True)
    pocs_n_iter               = 10,
    pocs_f_min_bin            = 2,
    pocs_f_max_bin            = 100,
)

result = run_experiment(hp_single)
print(f'Result: {result["run_label"]} → {result["best_val_acc"]:.1f}%')

BASE_HP = dict(
    pool_output_size          = None,
    hidden_layers_size        = [128, 64],
    activation                = 'relu',
    kernel_size               = (5, 5),
    filters                   = [8, 16, 32, 32, 32],
    batch_normalization       = True,
    dropout_rate              = 0.3,
    learning_rate             = 1e-4,
    max_epoch                 = 50,
    seed                      = 42,
    ae_base_channels          = 16,
    ae_learning_rate          = 1e-3,
    ae_max_epoch              = 20,
    pocs_n_iter               = 10,
    pocs_f_min_bin            = 2,
    pocs_f_max_bin            = 100,
    spec_augment_freq_mask    = 25,
    spec_augment_time_mask    = 50,
    spec_augment_n_freq_masks = 3,
    spec_augment_n_time_masks = 3,
    mixup_alpha               = 0.4,
)

RUNS = [
    # Reference: pool only
    dict(pool_every_stage=True,  use_se_block=False, use_tf_attention=False,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=False, use_mixup=False),
    # Pool + attention blocks
    dict(pool_every_stage=True,  use_se_block=True,  use_tf_attention=True,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=False, use_mixup=False),
    # Pool + attention + AutoEncoder
    dict(pool_every_stage=True,  use_se_block=True,  use_tf_attention=True,
         use_autoencoder=True,   use_pocs=False, use_spec_augment=False, use_mixup=False),
    # Pool + SpecAugment only
    dict(pool_every_stage=True,  use_se_block=False, use_tf_attention=False,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=True,  use_mixup=False),
    # Pool + Mixup only
    dict(pool_every_stage=True,  use_se_block=False, use_tf_attention=False,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=False, use_mixup=True),
    # Pool + SpecAugment + Mixup
    dict(pool_every_stage=True,  use_se_block=False, use_tf_attention=False,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=True,  use_mixup=True),
    # Pool + attention + SpecAugment
    dict(pool_every_stage=True,  use_se_block=True,  use_tf_attention=True,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=True,  use_mixup=False),
    # Pool + attention + Mixup
    dict(pool_every_stage=True,  use_se_block=True,  use_tf_attention=True,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=False, use_mixup=True),
    # Pool + attention + SpecAugment + Mixup 
    dict(pool_every_stage=True,  use_se_block=True,  use_tf_attention=True,
         use_autoencoder=False,  use_pocs=False, use_spec_augment=True,  use_mixup=True),
]

results = []
for i, flags in enumerate(RUNS):
    hp     = {**BASE_HP, **flags}
    result = run_experiment(hp)
    results.append(result)
    print(f'[R{i+1}] {result["run_label"]} → {result["best_val_acc"]:.1f}%')

print('\n' + '='*65)
print('SUMMARY')
print('='*65)
for i, r in enumerate(results):
    print(f'R{i+1:2d}  {r["best_val_acc"]:5.1f}%   {r["run_label"]}')
print('='*65)
best = max(results, key=lambda r: r['best_val_acc'])
print(f'Best : {best["run_label"]} → {best["best_val_acc"]:.1f}%')
