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
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f'Project root  : {PROJECT_ROOT}')
print(f'Checkpoints   : {CHECKPOINT_DIR}')
print(f'Results       : {RESULTS_DIR}')

if USE_COLAB:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        'torchaudio', 'soundfile', 'scipy', 'torchinfo', 'scikit-learn',
    ])

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

# Load test split
x_test_raw = torch.load(os.path.join(SPEC_DIR, 'test_data_batches.pt'))
y_test_raw = torch.load(os.path.join(SPEC_DIR, 'test_label_batches.pt'))
classes    = open(os.path.join(SPEC_DIR, 'classes.txt')).read().strip().split('\n')

x_all = torch.cat(x_test_raw)
y_all = torch.cat(y_test_raw)
x_test = [x_all[i:i+32] for i in range(0, len(x_all), 32)]
y_test = [y_all[i:i+32] for i in range(0, len(y_all), 32)]

print(f'Test samples : {len(y_all)}  |  Classes : {len(classes)}')

import json, glob

metrics_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'metrics_*.json')))
print(f'Found {len(metrics_files)} checkpoint(s).\n')
print(f'{"Timestamp":<30}  {"Best val":>8}  Run label')
print('-' * 80)
for path in metrics_files:
    with open(path) as f:
        log = json.load(f)
    ts       = path.split('metrics_')[-1].replace('.json', '')
    best_val = log.get('best_val_acc', max(log['valid_accuracy_list']))
    label    = log.get('run_label', 'unknown')
    print(f'{ts:<30}  {best_val:>7.1f}%  {label}')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ClassesML.ResNetImprovedV2 import ResNetImprovedV2
from ClassesML.ConvAutoEncoder import ConvAutoEncoder
from ClassesML.POCS import pocs_postprocess

# ── Select checkpoint ──────────────────────────────────────────────────────
TARGET_TIMESTAMP = '20260622_233155'   # paste timestamp here
# ──────────────────────────────────────────────────────────────────────────

metrics_path    = os.path.join(CHECKPOINT_DIR, f'metrics_{TARGET_TIMESTAMP}.json')
classifier_path = os.path.join(CHECKPOINT_DIR, f'resnet_{TARGET_TIMESTAMP}.pt')

if not os.path.exists(metrics_path) or not os.path.exists(classifier_path):
    raise FileNotFoundError(f'Checkpoint {TARGET_TIMESTAMP} not found. Run cell 2 to list available runs.')

with open(metrics_path) as f:
    logs = json.load(f)

hp        = logs['hyperparameter']
run_label = logs.get('run_label', TARGET_TIMESTAMP)
hp['input_dim']   = tuple(hp['input_dim'])
hp['kernel_size'] = tuple(hp['kernel_size'])

print(f'Run   : {run_label}')
print(f'Best val (training) : {logs["best_val_acc"]:.1f}% @ epoch {logs["best_epoch"]}')

# Rebuild model and load weights
model = ResNetImprovedV2(hp)
model.load_state_dict(torch.load(classifier_path, map_location=device))
model.to(device)
model.eval()

# Reload AE if used
ae_model = None
if hp.get('use_autoencoder'):
    ae_path = logs.get('ae_weights_path') or os.path.join(CHECKPOINT_DIR, f'autoencoder_{TARGET_TIMESTAMP}.pt')
    if os.path.exists(ae_path):
        ae_model = ConvAutoEncoder(hp)
        ae_model.load_state_dict(torch.load(ae_path, map_location=device))
        ae_model.to(device)
        ae_model.eval()
    else:
        print('WARNING: AE weights not found — skipping AE preprocessing.')

# Inference on test set
all_preds, all_true = [], []
with torch.no_grad():
    for x, y in zip(x_test, y_test):
        x = x.to(device)
        if ae_model is not None:
            x = ae_model(x)
        if hp.get('use_pocs'):
            x = pocs_postprocess(x, hp)
        all_preds.append(torch.argmax(model(x), dim=1).cpu().numpy())
        all_true.append(y.numpy())

pred_labels = np.concatenate(all_preds)
true_labels = np.concatenate(all_true)
test_acc    = (pred_labels == true_labels).mean() * 100

print(f'\nTest Accuracy : {test_acc:.2f}%')

# Per-class accuracy
print('\nPer-class accuracy:')
for i, cls in enumerate(classes):
    mask = true_labels == i
    acc  = (pred_labels[mask] == true_labels[mask]).mean() * 100 if mask.sum() > 0 else 0
    print(f'  {cls:<35} : {acc:.1f}%')

# Confusion matrix
cm   = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(13, 11))
disp.plot(ax=ax, xticks_rotation='vertical', cmap=plt.cm.Blues)
ax.set_title(f'{run_label}\nTest Accuracy: {test_acc:.2f}%', pad=20, fontsize=10)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{TARGET_TIMESTAMP}.png')
fig.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved : {cm_path}')

# Learning curves
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(logs['train_accuracy_list'], 'b-', linewidth=2, label='Train')
ax.plot(logs['valid_accuracy_list'], 'r-', linewidth=2, label='Validation')
best_ep = logs['best_epoch']
ax.axvline(x=best_ep - 1, color='green', linestyle='--', alpha=0.7,
           label=f'Best val: {logs["best_val_acc"]:.1f}% @ epoch {best_ep}')
ax.set_title(f'{run_label}\n{TARGET_TIMESTAMP}', fontsize=10)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.legend()
ax.grid(True)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(RESULTS_DIR, f'learning_curves_{TARGET_TIMESTAMP}.png'), dpi=150, bbox_inches='tight')
plt.show()

import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ClassesML.ResNetImprovedV2 import ResNetImprovedV2
from ClassesML.ConvAutoEncoder import ConvAutoEncoder
from ClassesML.POCS import pocs_postprocess

metrics_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'metrics_*.json')))
print(f'Evaluating {len(metrics_files)} checkpoint(s) on the test set...')

eval_results = []

for path in metrics_files:
    ts = path.split('metrics_')[-1].replace('.json', '')
    classifier_path = os.path.join(CHECKPOINT_DIR, f'resnet_{ts}.pt')

    if not os.path.exists(classifier_path):
        print(f'[SKIP] {ts} — weights not found')
        continue

    with open(path) as f:
        logs = json.load(f)

    hp        = logs['hyperparameter']
    run_label = logs.get('run_label', ts)
    hp['input_dim']   = tuple(hp['input_dim'])
    hp['kernel_size'] = tuple(hp['kernel_size'])

    model = ResNetImprovedV2(hp)
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    model.to(device)
    model.eval()

    ae_model = None
    if hp.get('use_autoencoder'):
        ae_path = logs.get('ae_weights_path') or os.path.join(CHECKPOINT_DIR, f'autoencoder_{ts}.pt')
        if os.path.exists(ae_path):
            ae_model = ConvAutoEncoder(hp)
            ae_model.load_state_dict(torch.load(ae_path, map_location=device))
            ae_model.to(device)
            ae_model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in zip(x_test, y_test):
            x = x.to(device)
            if ae_model is not None:
                x = ae_model(x)
            if hp.get('use_pocs'):
                x = pocs_postprocess(x, hp)
            all_preds.append(torch.argmax(model(x), dim=1).cpu().numpy())
            all_true.append(y.numpy())

    pred_labels = np.concatenate(all_preds)
    true_labels = np.concatenate(all_true)
    test_acc    = (pred_labels == true_labels).mean() * 100

    eval_results.append({'run_label': run_label, 'test_acc': test_acc, 'timestamp': ts})
    print(f'  {run_label:<55} : {test_acc:.1f}%')

eval_results.sort(key=lambda r: r['test_acc'], reverse=True)

# Comparison bar chart
labels   = [r['run_label'] for r in eval_results]
test_acc = [r['test_acc']  for r in eval_results]

fig, ax = plt.subplots(figsize=(14, max(4, len(labels) * 0.55)))
bars = ax.barh(range(len(labels)), test_acc, color='steelblue', alpha=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Test Accuracy (%)')
ax.set_title('All runs — Test Set Accuracy')
ax.set_xlim(0, 100)
for bar, acc in zip(bars, test_acc):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{acc:.1f}%', va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'comparison_test_accuracy.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nFigure saved : {out_path}')

print('\n' + '='*72)
print(f'{"Run label":<55}  Test acc  Timestamp')
print('='*72)
for r in eval_results:
    print(f'{r["run_label"]:<55}  {r["test_acc"]:>6.1f}%   {r["timestamp"]}')
print('='*72)
