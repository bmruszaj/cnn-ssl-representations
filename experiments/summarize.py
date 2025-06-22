import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from src.utils.params_yaml import load_yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()

def collect_accuracies():

    result_dir = PROJECT_ROOT / params['paths']['results_dir']
    result_dir.mkdir(parents=True, exist_ok=True)

    accs = {}
    for filename in os.listdir(result_dir):
        if filename.lower().endswith('.json'):

            file_path = result_dir / filename

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            acc = data.get('accuracy', None)
            if acc is not None:
                accs[filename.lower().split('_acc')[0]] = acc

    return accs

def plot_bar(acc):

    plt_dir = PROJECT_ROOT / params['paths']['plots_dir']
    plt_dir.mkdir(parents=True, exist_ok=True)

    if acc:
        names = list(acc.keys())
        values = [acc[n] for n in names]

        plt.figure(figsize=(10, 6))
        plt.bar(names, values)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Accuracy on test dataset')
        plt.tight_layout()
        plt.savefig(plt_dir / 'test_acc.png')
        plt.close()

if __name__ == '__main__':
    accs = collect_accuracies()
    plot_bar(accs)