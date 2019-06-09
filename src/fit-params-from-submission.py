import multiprocessing as mp
from datetime import datetime

import click
import numpy as np
from sklearn.metrics import r2_score

from fit_params import FitParamsThreadParams, load_data, FitParamsThreadResult, \
    FitParamsSamplesResult
from genetic import fit_svr

np.random.seed(420)

params_dict = {
    'kernel': ['rbf'],
    'gamma': [1 / i for i in range(80, 130, 20)],
    'C': [1e0, 1e1, 1e2, 1e3],
    'epsilon': [2e-2, 4e-2, 8e-2],
    'shrinking': [False]
}


def fit_params_for_samples(
        samples: np.array,
        params: FitParamsThreadParams
) -> FitParamsSamplesResult:
    model = fit_svr(
        params.train_data.X[samples],
        params.train_data.y[samples],
        params.valid_data.X,
        params.valid_data.y,
        n_iter=params.n_iter,
        params_dict=params_dict
    )
    model_params = model.get_params()
    model_score = r2_score(
        params.valid_data.y,
        model.predict(params.valid_data.X)
    )
    return FitParamsSamplesResult(
        samples=samples,
        epsilon=model_params['epsilon'],
        C=model_params['C'],
        gamma=model_params['gamma'],
        score=model_score
    )


def fit_params_thread(params: FitParamsThreadParams) -> FitParamsThreadResult:
    best_score_model_result = fit_params_for_samples(params.genetic_results, params)
    return FitParamsThreadResult(
        dataset_size=params.dataset_size,
        best_score_model_result=best_score_model_result,
    )


@click.command()
@click.argument("input-path", type=str)
@click.option("-n", "--n-threads", default=4)
@click.option("-i", "--input-dir", type=str, default='../data')
@click.option("-s", "--output-path", type=str,
              default=datetime.now().strftime("../data/fitted-submission-%d-%m-%y-%H-%M-%S.txt"))
def main(n_threads, input_path, input_dir, output_path):
    genetic_results = {}
    dataset_sizes = range(600, 1600, 100)
    with open(input_path, 'r') as input_file:
        for line, dataset_size in zip(input_file, dataset_sizes):
            _, _, _, samples_string = line.split(";")
            genetic_results[dataset_size] = [int(sample) for sample in samples_string.split(",")]

    train_data, valid_data = load_data(input_dir)
    fit_params = [
        FitParamsThreadParams(
            dataset_size=dataset_size,
            genetic_results=genetic_results[dataset_size],
            train_data=train_data,
            valid_data=valid_data,
            n_iter=24
        ) for dataset_size in dataset_sizes
    ]
    print(f"Fitting parameters...")
    with mp.Pool(n_threads) as pool:
        results = pool.map(fit_params_thread, fit_params)

    best_score_model_results = [result.best_score_model_result for result in results]
    with open(output_path, 'w') as output_file:
        for result in best_score_model_results:
            normalized_samples = np.array(result.samples) + 1
            samples_string = ",".join([str(sample) for sample in normalized_samples])
            print(f"Dataset size: {len(result.samples)}, score: {result.score}")
            output_file.write(f"{result.epsilon};{result.C};{result.gamma};{samples_string}\n")
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
