# Setup
1. Download the datasets from https://www.kaggle.com/datasets/patricktakenaka/vipro-2-datasets
2. Unpack the datasets 
3. Set the environment variable DATA_DIR in the `.env` of this repository to the parent folder of the datasets
4. Create and activate the conda environment
```
conda env create --file environment.yml
conda activate vipro2
```


# Training
Start the training with
```
python3 train.py --config=psf/configs/orbits-2d.py --random_seed 41 --deterministic=True --run_name orbits-2d
```

or

```
python3 train.py --config=psf/configs/orbits-3d.py --random_seed 41 --deterministic=True --run_name orbits-3d
```

Wandb is used for logging the results.


