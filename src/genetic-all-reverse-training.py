import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import NamedTuple, List
from itertools import repeat

from tqdm import trange
import click
import numpy as np
from sklearn.model_selection import train_test_split

from genetic import GenerationResult, DataSet, EvolutionParams, run_evolution, fit_svr

np.random.seed(420)


def best_matching_samples(
        train_X: np.array,
        train_y: np.array,
        valid_X: np.array,
        valid_y: np.array,
        sample_ids: np.array,
        weights: np.array,
        n_iter: int,
        n_samples: int=1600,
        n_reverse_train_samples: int=2000
) -> np.array:
    """
    Returns ids of n_samples that are most accurately predicted
    by SVR fitted on validation subset of n_reverse_train_samples.
    """
    train_ids = np.random.choice(
        sample_ids,
        size=(3* n_reverse_train_samples),
        replace=False,
        p=weights
    )
    Xrev_train, Xrev_test, yrev_train, yrev_test = train_test_split(
        valid_X, valid_y, train_size=n_reverse_train_samples
    )
    # evaluate model on a subset of validation set + the some samples from the training set
    X_full_test = np.vstack((train_X[train_ids], Xrev_test))
    y_full_test = np.concatenate((train_y[train_ids], yrev_test))
    model = fit_svr(Xrev_train, yrev_train, X_full_test, y_full_test, n_iter=n_iter)
    y_pred = model.predict(train_X)
    mse = (y_pred-train_y)**2
    best_samples = np.argsort(mse)[:n_samples]
    return best_samples


class ReverseMatchingParams(NamedTuple):
    train_data: DataSet
    valid_data: DataSet
    evolution_params: EvolutionParams
    n_samples: int
    n_reverse_train_samples: int


def best_matching_thread(
        params: ReverseMatchingParams
) -> np.array:
    return best_matching_samples(
        params.train_data.X,
        params.train_data.y,
        params.valid_data.X,
        params.valid_data.y,
        params.train_data.ids,
        params.evolution_params.weights,
        n_iter = params.evolution_params.n_fits,
        n_samples = params.n_samples,
        n_reverse_train_samples = params.n_reverse_train_samples
    )


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
@click.option("-n", "--n-threads", default=4)
@click.option("-i", "--input-dir", type=str, default='../data')
@click.option("-o", "--output-path", type=str,
              default=datetime.now().strftime('../data/genetic-all-%d-%m-%y-%H-%M-%S.pkl'))
@click.option("-s", "--submission-path", type=str,
              default=datetime.now().strftime('../data/sumbission-%d-%m-%y-%H-%M-%S.txt'))
def main(
        n_threads,
        input_dir,
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
    weights = np.load(input_dir / 'train_nofGames.npy')
    weights[weights < weights.mean()] = 0
    weights = weights / weights.sum()
    params = EvolutionParams(
        n_models=32,
        n_fits=12,
        n_generations=64,
        n_train_samples=1500,
        n_valid_samples=6000,
        train_ids=None,
        mutation_prob=0.4,
        score_mode="weights",
        weights=weights,
    )
    rmp = ReverseMatchingParams(
        train_data,
        valid_data,
        params,
        n_samples=1600,
        n_reverse_train_samples=2000
    )

    results = {}
    with mp.Pool(n_threads) as pool:
        print("Fitting models on validation data...")
        final_model_samples = np.array(pool.map(
            best_matching_thread,
            repeat(rmp, params.n_models)
        ))

        with trange(1500, 500, -100) as t:
            for dataset_size in t:
                t.set_description(f"Dataset {dataset_size}")
                start_model_samples = shrink_samples(final_model_samples, dataset_size)
                assert start_model_samples.shape == (final_model_samples.shape[0], dataset_size)
                params = params._replace(n_train_samples=dataset_size, train_ids=start_model_samples)
                results[dataset_size] = run_evolution(train_data, valid_data, pool, params)
                best_gen = np.argmax(np.asarray([gen.model_scores.mean() for gen in results[dataset_size]]))
                final_model_samples = results[dataset_size][best_gen].model_samples
        print(f"Saving results to {output_path}...")
        pickle.dump(results, open(output_path, 'wb'))
        print(f"Saving sumbission to {submission_path}...")
        prepare_submission(results, submission_path)
        print("Done")


if __name__ == "__main__":
    main()
