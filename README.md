# clash-royale-challenge

## How to use `validation_mode`
Before we start you should know that there are three validaton sets in this project:
* `valid_*.npy` - these are original
* `reduced_valid_*.npy` - there are original samples but only 2/3 most frequently played from original datset
* `valid_mix_*.npy` - this is concatenated `reduced_valid` with 4000 most frequently played desc from train dataset

### `original` validation mode
In this mode program takes supplied validation dataset, and samples from it `n_valid_samples`. These are used to validate all models in whole evolution process.

### `upsampling` validation mode
In this mode **at each generation** program takes supplied validation dataset and doubbles its length by upsampling from most freqently played decs from train dataset (every generation new samples). Then it samples from extended validation dataset `n_valid_samples`. These are used to validate models in this generation.

### Experiments
To run experiment **VARIANT 1** use `original` and `valid_mix`

To run experiment **VARIANT 2** use `upsampling` and `reduced_val`