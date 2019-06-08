import numpy as np
import multiprocessing as mp

import pickle
import click

from multiprocessing.pool import Pool
from typing import Dict, List, NamedTuple
from pathlib import Path
from datetime import datetime

from genetic import DataSet, fit_svr, GenerationResult

np.random.seed(420)

params_dict = {
    'kernel': ['rbf'],
    'gamma': [1 / i for i in range(80, 130, 10)],
    'C': [0.9, 1.0, 1.1],
    'epsilon': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    'shrinking': [True]
}


class FitParamsThreadParams(NamedTuple):
    dataset_size: int
    genetic_results: List[GenerationResult]
    train_data: DataSet
    valid_data: DataSet
    n_iter: int


class FitParamsThreadResult(NamedTuple):
    dataset_size: int
    epsilon: float
    C: float
    gamma: float
    samples: np.array


def find_best_model_samples(results):
    model_scores = np.concatenate([result.model_scores for result in results])
    model_samples = np.concatenate([result.model_samples.astype('int64') for result in results])
    best_model_id = np.argmax(model_scores)
    best_model_samples = model_samples[best_model_id]
    return best_model_samples


def fit_params_thread(params: FitParamsThreadParams) -> FitParamsThreadResult:
    best_samples = find_best_model_samples(params.genetic_results)
    best_params = fit_svr(
        params.train_data.X[best_samples],
        params.train_data.y[best_samples],
        params.valid_data.X,
        params.valid_data.y,
        n_iter=params.n_iter,
        params_dict=params_dict
    ).get_params()
    return FitParamsThreadResult(
        dataset_size=params.dataset_size,
        epsilon=best_params['epsilon'],
        C=best_params['C'],
        gamma=best_params['gamma'],
        samples=best_samples
    )


def create_submission(
        train_data: DataSet,
        valid_data: DataSet,
        genetic_results: Dict[int, List[GenerationResult]],
        output_dir: str,
        pool: Pool
):
    dataset_sizes = sorted(genetic_results.keys())
    fit_params = [
        FitParamsThreadParams(
            dataset_size=dataset_size,
            genetic_results=genetic_results[dataset_size],
            train_data=train_data,
            valid_data=valid_data,
            n_iter=32
        ) for dataset_size in dataset_sizes
    ]
    print(f"Fitting parameters...")
    results = pool.map(fit_params_thread, fit_params)
    file_path = Path(output_dir) / datetime.now().strftime('submission-%d-%m-%y-%H-%M-%S.txt')
    with open(file_path, 'w') as file:
        for result in results:
            print(f"size: {result.dataset_size}, "
                  f"epsilon: {result.epsilon}, C: {result.C}, gamma: {result.gamma}")
            samples_string = ",".join([str(sample) for sample in result.samples])
            file.write(f"{result.epsilon};{result.C};{result.gamma};{samples_string}\n")
    print(f"Saved submission to {file_path}...")


def load_data(input_dir: str):
    input_dir = Path(input_dir).resolve()
    train_X = np.load(input_dir / 'train_X.npy')
    train_y = np.load(input_dir / 'train_y.npy')
    valid_X = np.load(input_dir / 'valid_X.npy')
    valid_y = np.load(input_dir / 'valid_y.npy')
    train_data = DataSet(train_X, train_y, np.arange(len(train_X)))
    valid_data = DataSet(valid_X, valid_y, np.arange(len(valid_X)) * (-1))
    return train_data, valid_data


@click.command()
@click.argument("input-file", type=str)
@click.option("-n", "--n-threads", default=4)
@click.option("-i", "--input-dir", type=str, default='../data')
@click.option("-s", "--output-dir", type=str, default='../data/')
def main(n_threads, input_file, input_dir, output_dir):
    train_data, valid_data = load_data(input_dir)
    with open(input_file, 'rb') as file:
        genetic_results = pickle.load(file)
    with mp.Pool(n_threads) as pool:
        create_submission(train_data, valid_data, genetic_results, output_dir, pool)


if __name__ == "__main__":
    main()
