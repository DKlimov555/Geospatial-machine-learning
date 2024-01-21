# Training Geospatial models

This is a replication and application of Combining satellite imagery and machine learning to predict poverty (Jean et al. 2016). I'm closely following jmathur25's replication guide, found here: https://github.com/jmathur25/predicting-poverty-replication/tree/master

All scripts are put in modifiable Jupyter Notebook (.ipynb)

I've tried very hard to bugfix the jmather25's script and getting all functions using outdated modules to run.
For comprehension (both mine and any future user's), I've also made a ton of comments.

I recommend creating a virtual environment before starting, such as miniconda.

First run:

```
conda create -n <ENV_NAME> python=3.7 pip gdal
conda activate <ENV_NAME>
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

If you want to run Jupyter Notebooks in an environment, run the following inside the environment:

```
pip install --user ipykernel
python -m ipykernel install --user --name=<ENV_NAME>
```

Then, set the kernel for all the Jupyter files to whatever your ENV_NAME is.

To allow tqdm (the progress bar library) to run in a Jupyter Notebook, also run:

```
conda install -c conda-forge ipywidgets
```

If you just want to run the model and get the plots:
    here's the entire setup for 2015: https://1drv.ms/f/s!AvvtTfGA0Ya8srItpqhP2C0mqCs0pg?e=7o3HSD 
    and for 2019: https://1drv.ms/f/s!AvvtTfGA0Ya8ts9VextT6_lCmHFBMw?e=1bJojd

Run:
```
scripts/feature_extract.ipynb
scripts/predict_consumption.ipynb
```

If you want to train your own model, an NVIDIA GPU is more or less required. AMD is also viable under Linux.

```
git clone https://github.com/Quakespeare/Geospatial-machine-learning
```

1. Get the nightlight images for 2015 here: https://drive.google.com/drive/folders/1gZZ1NoKaq43znWIBjzmrLuMQh4uzu9qn?usp=sharing
And for 2019 here: https://eogdata.mines.edu/nighttime_light/annual/v20/2019/VNL_v2_npp_2019_global_vcmslcfg_c202102150000.average_masked.tif.gz
Put them im \data\nightlights

2. Get the LSMS survey data from the world bank. Download the 2016-2017 Malawi survey data, 2015-2016 Ethiopia data, and the 2015-2016 Nigeria data from https://microdata.worldbank.org/index.php/catalog/lsms. The World Bank wants to know how people use their data, so you will have to sign in and explain why you want their data. Make sure to download the CSV version. Unzip the downloaded data into `countries/<country name>/LSMS/`. Country name should be either `malawi_2016`, `ethiopia_2015`, or `nigeria_2015`.

Alternatively, follow the Onedrive links above and copy \data\countries\<country name>\LSMS


Run the Jupyter files in the following order:

```
scripts/process_survey_data.ipynb
scripts/download_images.ipynb
scripts/train_cnn.ipynb
scripts/feature_extract.ipynb
scripts/predict_consumption.ipynb
```

Even if you choose to do that, i recommend getting the daytime satellite image files from the Onedrive links above. You find them under \data\nightlights and \data\<country name>\images. Heads up: It's A LOT of data.
