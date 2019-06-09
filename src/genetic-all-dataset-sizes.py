import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import NamedTuple, List

from tqdm import trange
import click
import numpy as np

from genetic import GenerationResult, DataSet, EvolutionParams, run_evolution

np.random.seed(420)


def shrink_samples(samples: np.array, size: int) -> np.array:
    return np.array([
        np.random.choice(sample, size, replace=False).astype('int64')
        for sample in samples
    ])


def prepare_submission(results, path):
    with open(path, 'w') as file:
        for dataset_size in sorted(results.keys()):
            dataset_results = results[dataset_size]
            model_scores = np.concatenate([result.model_scores for result in dataset_results])
            model_samples = np.concatenate([result.model_samples.astype('int64') for result in dataset_results])
            model_params = np.concatenate([result.model_params for result in dataset_results])
            best_model_id = np.argmax(model_scores)
            best_model_params = model_params[best_model_id]
            epsilon, C, gamma = [best_model_params.get(key) for key in ['epsilon', 'C', 'gamma']]
            best_model_sample = model_samples[best_model_id]
            best_model_sample_string = ",".join([str(idx) for idx in best_model_sample])
            file.write(f"{epsilon};{C};{gamma};{best_model_sample_string}\n")


@click.command()
@click.argument("input-file", type=str)
@click.option("-n", "--n-threads", default=4)
@click.option("-i", "--input-dir", type=str, default='../data')
@click.option("-o", "--output-path", type=str,
              default=datetime.now().strftime('../data/genetic-%d-%m-%y-%H-%M-%S.pkl'))
@click.option("-s", "--submission-path", type=str,
              default=datetime.now().strftime('../data/sumbission-%d-%m-%y-%H-%M-%S.txt'))
def main(
        n_threads,
        input_dir,
        input_file,
        output_path,
        submission_path
):
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()
    train_X = np.load(input_dir / 'train_X.npy')
    train_y = np.load(input_dir / 'train_y.npy')
    valid_X = np.load(input_dir / 'valid_X.npy')
    valid_y = np.load(input_dir / 'valid_y.npy')
    train_data = DataSet(train_X, train_y, np.arange(len(train_X)))
    valid_data = DataSet(valid_X, valid_y, np.arange(len(valid_X)) * (-1))
    weights = np.load("../data/train_nofGames.npy")
    weights[weights < weights.mean()] = 0
    weights = weights / weights.sum()
    # weights = np.log(weights)
    # assert weights.min() >= 1
    # weights = weights / weights.max()

    params = EvolutionParams(
        n_models = 24,
        n_fits = 9,
        n_generations = 16,
        n_train_samples = 1500,
        n_valid_samples = 6000,
        train_ids = None,
        mutation_prob = 0.04,
        score_mode = "weights",
        weights = weights,
    )

    results = {}
    with open(input_file, 'rb') as file:
        results[1500] = pickle.load(file)
        final_model_samples = results[1500][-1].model_samples

    with mp.Pool(n_threads) as pool:
        with trange(1400, 500, -100) as t:
            for dataset_size in t:
                t.set_description(f"Dataset {dataset_size}")
                start_model_samples = shrink_samples(final_model_samples, dataset_size)
                assert start_model_samples.shape == (final_model_samples.shape[0], dataset_size)
                params = params._replace(n_train_samples=dataset_size, train_ids=start_model_samples)
                results[dataset_size] = run_evolution(train_data, valid_data, pool, params)
                final_model_samples = results[dataset_size][-1].model_samples
        print(f"Saving results to {output_path}...")
        pickle.dump(results, open(output_path, 'wb'))
        print(f"Saving sumbission to {submission_path}...")
        prepare_submission(results, submission_path)
        print("Done")


if __name__ == "__main__":
    main()
