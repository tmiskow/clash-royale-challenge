"""
3rd (and last) script in sequence,
tuning model parameters from larger grid
and generating various submission scenarios.
"""

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
    'gamma': [1 / 90],
    'C': [1e0, 1e1],
    'epsilon': [0.02],
    'shrinking': [False]
}


class FitParamsThreadParams(NamedTuple):
    dataset_size: int
    genetic_results: List[GenerationResult]
    train_data: DataSet
    valid_data: DataSet
    n_iter: int


class FitParamsSamplesResult(NamedTuple):
    samples: np.array
    epsilon: float
    C: float
    gamma: float


class FitParamsThreadResult(NamedTuple):
    dataset_size: int
    best_score_model_result: FitParamsSamplesResult
    worst_score_model_result: FitParamsSamplesResult
    best_frequency_model_result: FitParamsSamplesResult
    most_frequent_samples_result: FitParamsSamplesResult


def get_best_score_model_samples(results: List[GenerationResult]) -> np.array:
    model_scores = np.concatenate([result.model_scores for result in results])
    model_samples = np.concatenate([result.model_samples.astype('int64') for result in results])
    best_model_id = np.argmax(model_scores)
    best_model_samples = model_samples[best_model_id]
    return best_model_samples


def get_worst_score_model_samples(results: List[GenerationResult]) -> np.array:
    model_scores = np.concatenate([result.model_scores for result in results])
    model_samples = np.concatenate([result.model_samples.astype('int64') for result in results])
    worst_model_id = np.argmin(model_scores)
    worst_model_samples = model_samples[worst_model_id]
    return worst_model_samples


# create new dataset from most frequently appearing samples
def get_most_frequent_samples(results: List[GenerationResult], size: int) -> np.array:
    samples = np.concatenate(np.concatenate(
        [result.model_samples.astype('int64') for result in results]))
    unique_samples, unique_samples_counts = np.unique(samples, return_counts=True)
    most_frequent_ids = np.argsort(unique_samples_counts)[:size]
    most_frequent_samples = unique_samples[most_frequent_ids]
    return most_frequent_samples


#  select dataset that has largest number of frequently appearing samples
def get_best_frequency_model_samples(results: List[GenerationResult]) -> np.array:
    model_samples = np.concatenate([result.model_samples.astype('int64') for result in results])
    samples = np.concatenate(model_samples)
    n_models = len(model_samples)
    _, unique_inverse, unique_counts = np.unique(samples, return_inverse=True, return_counts=True)
    samples_counts = unique_counts[unique_inverse]
    assert len(samples) == len(samples_counts)
    model_samples_counts = np.split(samples_counts, n_models)
    model_frequency_scores = np.sum(model_samples_counts, axis=1)
    best_frequency_model_id = np.argmax(model_frequency_scores)
    best_frequency_model_samples = model_samples[best_frequency_model_id]
    return best_frequency_model_samples


def fit_params_for_samples(
        samples: np.array,
        params: FitParamsThreadParams
) -> FitParamsSamplesResult:
    fitted_params = fit_svr(
        params.train_data.X[samples],
        params.train_data.y[samples],
        params.valid_data.X,
        params.valid_data.y,
        n_iter=params.n_iter,
        params_dict=params_dict
    ).get_params()
    return FitParamsSamplesResult(
        samples=samples,
        epsilon=fitted_params['epsilon'],
        C=fitted_params['C'],
        gamma=fitted_params['gamma']
    )


def fit_params_thread(params: FitParamsThreadParams) -> FitParamsThreadResult:
    best_frequency_model_result = fit_params_for_samples(
        get_best_frequency_model_samples(params.genetic_results), params)
    best_score_model_result = fit_params_for_samples(
        get_best_score_model_samples(params.genetic_results), params)
    worst_score_model_result = fit_params_for_samples(
        get_worst_score_model_samples(params.genetic_results), params)
    most_frequent_samples_result = fit_params_for_samples(
        get_most_frequent_samples(params.genetic_results, params.dataset_size), params)
    return FitParamsThreadResult(
        dataset_size=params.dataset_size,
        best_score_model_result=best_score_model_result,
        worst_score_model_result=worst_score_model_result,
        most_frequent_samples_result=most_frequent_samples_result,
        best_frequency_model_result=best_frequency_model_result
    )


def save_submission(results: List[FitParamsSamplesResult], output_dir: str, filename: str):
    path = Path(output_dir) / filename
    with open(path, "w") as file:
        for result in results:
            samples_string = ",".join([str(sample) for sample in result.samples])
            file.write(f"{result.epsilon};{result.C};{result.gamma};{samples_string}\n")
    print(f"Saved submission to {path}")


def save_results(results: List[FitParamsThreadResult], output_dir: str):
    base_filename = datetime.now().strftime('submission-%d-%m-%y-%H-%M-%S.txt')
    best_score_model_results = [result.best_score_model_result for result in results]
    save_submission(best_score_model_results, output_dir, f"best-model-{base_filename}")
    worst_score_model_results = [result.worst_score_model_result for result in results]
    save_submission(worst_score_model_results, output_dir, f"worst-model-{base_filename}")
    most_frequent_samples_results = [result.most_frequent_samples_result for result in results]
    save_submission(most_frequent_samples_results, output_dir, f"most-frequent-{base_filename}")
    best_frequency_model_results = [result.best_frequency_model_result for result in results]
    save_submission(best_frequency_model_results, output_dir, f"best-frequency-{base_filename}")


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
            n_iter=2
        ) for dataset_size in dataset_sizes
    ]
    print(f"Fitting parameters...")
    results = pool.map(fit_params_thread, fit_params)
    save_results(results, output_dir)
    print("Done")


def load_data(input_dir: str):
    input_dir = Path(input_dir).resolve()
    train_X = np.load(input_dir / 'train_X.npy')
    train_y = np.load(input_dir / 'train_y.npy')
    valid_X = np.load(input_dir / 'valid_mix_X.npy')
    valid_y = np.load(input_dir / 'valid_mix_y.npy')
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
