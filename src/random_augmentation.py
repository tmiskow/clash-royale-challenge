from pathlib import Path

import numpy as np
import click
from datetime import datetime

from fit_params import load_data
from tqdm import tqdm

np.random.seed(420)


@click.command()
@click.argument("input-path", type=str)
@click.option("-i", "--input-dir", type=str, default='../data')
@click.option("-s", "--output-path", type=str,
              default=datetime.now().strftime("../data/fixed-submission-%d-%m-%y-%H-%M-%S.txt"))
def main(input_path, input_dir, output_path):
    results = {}
    dataset_sizes = range(600, 1600, 100)
    with open(input_path, 'r') as input_file:
        for line, dataset_size in tqdm(zip(input_file, dataset_sizes)):
            epsilon, C, gamma, samples_string = line.split(";")
            samples = [int(sample)-1 for sample in samples_string.split(",")]
            results[dataset_size] = (epsilon, C, gamma, samples)

    train_data, valid_data = load_data(input_dir)
    weights = np.load(Path(input_dir) / 'train_nofGames.npy')
    weights[weights < weights.mean()] = 0
    weights = weights / weights.sum()
    with open(output_path, 'w') as output_file:
        for dataset_size in tqdm(dataset_sizes):
            epsilon, C, gamma, samples = results[dataset_size]
            bool_mask = [False if i in samples else True for i in train_data.ids]
            unused_samples = train_data.ids[bool_mask]
            random_ids = np.random.choice(dataset_size, int(0.05*dataset_size), replace=False).astype('int64')
            probs = weights[bool_mask] / weights[bool_mask].sum()
            randomly_selected_samples = np.random.choice(unused_samples, int(0.05*dataset_size), p=probs, replace=False).astype('int64')
            for i in range(len(randomly_selected_samples)):
                samples[random_ids[i]] = randomly_selected_samples[i]
            normalized_samples = np.array(samples) + 1
            samples_string = ",".join([str(sample) for sample in normalized_samples])
            output_file.write(f"{epsilon};{C};{gamma};{samples_string}\n")
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()