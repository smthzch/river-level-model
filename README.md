# River Level Model

This project provides the utilities to fetch USGS river camera imagery and build a Convolutional Neural Network to predict river level based on imagery.

Both ingest and training are managed by a `Metaflow`.

# Setup

TODO

# Ingest Data

Data is pulled from USGS webcams and the service seems to be down often.

Pull data with `python ingestflow.py run`

Each time this is run a single new image will be pulled and new stage data will be pulled for images if it is available.

I set up a cron job to run this every hour, it takes a while to build up a dataset worth anything.

# Training

Training is performed by calling `python trainflow.py run`

Call `python trainflow.py run --help` to view available command line parameters.

By default the model, metrics, and validation plots are saved to the `runs/run_{datetime.now()}` directory.
This directory is created if it does not exist.

# Results

![](figures/test_predictions.png)

![](figures/true_v_predicted.png)
