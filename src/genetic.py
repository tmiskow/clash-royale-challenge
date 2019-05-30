"""
Fast, parallelizable genetic algorithm implementation with some shortcuts.

Numpy array naming convention:
- *_ids = array of number corresponding to rows in the dataset
- *_index = boolean array allowing for fast selection from the dataset
"""

import pandas as pd
import numpy as np
np.random.seed(420)

from sklearn.svm import SVR
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import r2_score

from tqdm import trange
from numba import jit
import click

import multiprocessing as mp
import pickle
from itertools import repeat
from pathlib import Path
from datetime import datetime
from typing import NamedTuple, List, Tuple


class DataSet(NamedTuple):
    X: np.array  # (n_samples, n_features)
    y: np.array  # 1d
    ids: np.array  # 1d


class GenerationParams(NamedTuple):
    n_models: int  # = n datasets
    n_fits: int  # per each model during hyperparameter optimization
    train_data: DataSet
    n_train_samples: int
    train_probs: np.array
    valid_data: DataSet
    valid_index: np.array


class FitResult(NamedTuple):
    train_index: np.array
    sample_scores: np.array
    model_score: float
    model_params: dict


class GenerationResult(NamedTuple):
    train_probs: np.array
    train_index: np.array  # (n_samples), True for samples that were used in any of the models
    model_scores: np.array
    model_params: List[dict]
    model_samples: np.array  # (n_models, n_train_samples) - IDs, NOT INDEX


class EvolutionParams(NamedTuple):
    n_models: int  # = datasets per generation
    n_fits: int  # per each model during hyperparameter optimization
    n_generations: int
    n_train_samples: int
    n_valid_samples: int
    mutation_prob: float  # between 0 and 1


# defaults for random hyperparameter search
params_dict = {
    'kernel': ['rbf'],
    'gamma': [1 / i for i in range(80, 130, 10)],
    'C': [0.9, 1.0, 1.1],
    'epsilon': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    'shrinking': [True]
}


def sample(n_samples: int, ids: np.array, weights: np.array=None) -> np.array:
    selected_ids = np.random.choice(ids, n_samples, replace=False, p=weights)
    selected_index = np.isin(ids, selected_ids, assume_unique=True)
    return selected_index  # same shape as ids, for easier selection


def fit_svr(
        X_train: np.array,
        y_train: np.array,
        X_valid: np.array,
        y_valid: np.array,
        params_dict: dict=params_dict,
        n_iter: int=25
):
    ps = ParameterSampler(n_iter=n_iter, param_distributions=params_dict)
    scores = np.zeros(n_iter)
    models = list(repeat(None, n_iter))
    for idx, params in enumerate(ps):
        svr = SVR(**params)
        svr.fit(X_train, y_train)
        scores[idx] = r2_score(y_valid, svr.predict(X_valid))
        models[idx] = svr
    return models[np.argmax(scores)]


def fit_thread(params: GenerationParams) -> FitResult:
    train_index = sample(
        params.n_train_samples,
        params.train_data.ids,
        params.train_probs
    )  # TODO: Remove sampling from here and move to crossover_thread
    model = fit_svr(
        params.train_data.X[train_index],
        params.train_data.y[train_index],
        params.valid_data.X[params.valid_index],
        params.valid_data.y[params.valid_index],
        n_iter=params.n_fits
    )
    sample_scores = np.power(
        params.train_data.y - model.predict(params.train_data.X),
        2
    )
    model_score = r2_score(
        params.valid_data.y[params.valid_index],
        model.predict(params.valid_data.X[params.valid_index])
    )
    return FitResult(train_index, sample_scores, model_score, model.get_params())

def crossover_thread():
    pass  # TODO: Crossing and mutation, (Type: [np.array, np.array] -> np.array) - ONLY OPERATE ON INDICES

def run_generation(params: GenerationParams, n_models: int, pool: mp.Pool) -> Tuple[np.array, np.array, np.array, GenerationResult]:
    train_probs = np.zeros_like(params.train_probs)
    used_train_index = np.zeros_like(params.train_data.y, dtype=np.bool)
    model_params = list(repeat({}, params.n_models))
    model_scores = np.zeros(params.n_models)
    model_samples = np.zeros((params.n_models, params.n_train_samples))
    results = pool.map(fit_thread, repeat(params, n_models))
#     results = map(fit_thread, repeat(params, n_models))  # in case the Pool does not work in Jupyter
    for idx, fit_result in enumerate(results):
        used_train_index |= fit_result.train_index
        train_probs += fit_result.sample_scores * np.exp(fit_result.model_score)
        model_params[idx] = fit_result.model_params
        model_scores[idx] = fit_result.model_score
        model_samples[idx] = params.train_data.ids[fit_result.train_index].astype(np.uint)
    # TODO 1: Do the selection (by model fitness)
    # TODO 2: Generate pairs of fit datasets and use the same pool for parallel crossing
    # TODO 3: Return datasets for next generation, may need to change conents of GenerationResult
    return GenerationResult(train_probs, used_train_index, model_scores, model_params, model_samples)


def run_evolution(train_data: DataSet, valid_data: DataSet, pool: mp.Pool, params: EvolutionParams):
    valid_index = sample(params.n_valid_samples, valid_data.ids)
    train_probs = np.ones(len(train_data.ids)) / len(train_data.ids)
    results = []
    with trange(params.n_generations) as t:
        for generation_idx in t:
            t.set_description(f"Generation {generation_idx+1}")
            gen_results = run_generation(
                GenerationParams(
                    params.n_models,
                    params.n_fits,
                    train_data,
                    params.n_train_samples,
                    train_probs,
                    valid_data,
                    valid_index
                ),
                params.n_models,
                pool
            )
            # we simulate selecting samples for mutation by altering their probabilities:
            gen_train_probs = gen_results.train_probs
            gen_results.train_probs[gen_results.train_index] *= (1. - params.mutation_prob)
            gen_train_probs[~gen_results.train_index] *= params.mutation_prob
            train_probs += gen_train_probs
            train_probs /= sum(train_probs)
            results.append(gen_results)
            t.set_postfix(mean_score=sum(gen_results.model_scores)/len(gen_results.model_scores), max_score=max(gen_results.model_scores))
    return results


@click.command()
@click.option("-n", "--n-threads", default=4)
@click.option("-i", "--input-dir", type=str, default='../data')
@click.option("-o", "--output-path", type=str,
              default=datetime.now().strftime('../data/genetic-%d-%m-%y-%H-%M-%S.pkl'))
def main(n_threads, input_dir, output_path):
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()
    train_X = np.load(input_dir / 'train_X.npy')
    train_y = np.load(input_dir / 'train_y.npy')
    valid_X = np.load(input_dir / 'valid_X.npy')
    valid_y = np.load(input_dir / 'valid_y.npy')
    train_data = DataSet(train_X, train_y, np.arange(len(train_X)))
    valid_data = DataSet(valid_X, valid_y, np.arange(len(valid_X)) * (-1))
    params = EvolutionParams(
        n_models = 16,
        n_fits = 24,
        n_generations = 16,
        n_train_samples = 1500,
        n_valid_samples = 6000,
        mutation_prob = 0.16
    )
    with mp.Pool(n_threads) as pool:
        results = run_evolution(train_data, valid_data, pool, params)
        pickle.dump(results, open(output_path, 'wb'))

if __name__ == '__main__':
    main()
