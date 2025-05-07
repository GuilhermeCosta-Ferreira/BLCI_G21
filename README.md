# Brain Like Computation and Intelligence - Group 21

## Project Overview

This repository contains code and resources used to run the project of the BLCI course. The project explores task-driven and/or data driven models as well as a mix of the two to model and predict firing rates of IT neurons according to image datasets. In the different weeks(6,7,8), we utilize linear models, regularized ridge regression, PCA-Based dimensionality reduction, pre-trained and randomly initialized neural networks, as well as our own architectures and compare how well they perform.

## Repository Structure

The repository contains several files and folders:

- The main notebook has several portions in it: `week6`, `week7`, and `week8`:

- `week6` contains the week 6 assignment, where we run the task driven models as well as the initial linear and ridge regression models. There, it also refers to `my_neural_prediction_results` and `my_neural_prediction_results_randomly_init`

- `my_neural_prediction_results/` and `my_neural_prediction_results_randomly_init/`: These contain the pickle files for the randomly initialized model, as well as the results of the predictions

- `week7` contains a data-driven approach using a standard CNN, as well as its results

- `week8` contains also a python notebook with our best performing model's training, rationale, and testing. We used a pretrained TinySwin transformer branch in parallel with a ResNet50 pretrained branch, connected to a MLP head that performs predictions. THis is to encorporate both high level retinal features as well as global attention to the entire image to hopeully predict neuron activations

- `test_predictions.py` and `test.ipynb` contain what we need to test on new data, they rely on the model that we have trained and saved