# MachineLearningGeneralisedToricCodes
The abstract of the paper, _Machine Learning Generalised Toric Codes_:

We use machine learning techniques to approximate the minimum Hamming distance of generalised toric codes, leveraging the generator matrix of the toric code and the generator matrix of the dual code. Although the predictive accuracy is modest, the model effectively reduces the search space for generalised toric codes that achieve the greatest minimum Hamming distance for a fixed dimension. We pair this model with a genetic algorithm for constructing generalised toric codes, and discover new champion codes over $\mathbb{F}_8$. Our results show the use of machine learning with algorithmic search methods in the study and construction of error-correcting codes.

This repository includes all runs exploring $\mathbb{F}_7$, seen in the $\mathbb{F}_7$ folder, and all runs exploring $\mathbb{F}_8$, seen in the $\mathbb{F}_8$ folder. Both also include all code relating to the project.

# Structure

## Magma code

TBA

## Datasets

Contains $\mathbb{F}_7$ and $\mathbb{F}_8$ datasets as a text files as they were generated in Magma. Folders with `F_7` in the name contain datasets for $\mathbb{F}_7$ codes, with `F_8` - for $\mathbb{F}_8$. 

As explained in the paper, we generated two datasets for $\mathbb{F}_7$, initial one with 100,000 codes for all vertices - folder `dataset_10e5_F7_initial`, and the main one - `dataset_10e5_F7_by_n_vert`, with 100,000 codes for each generated number of vertices.

- `dataset_10e5_F7_initial` contains file `dataset_10e5_F7.txt.zip`, which is orgainised as a list, where each element is a list with the follwoing features: `[vertices in first order, vertices in second order, minimum hamming distance of toric code, first order approximation of hamming distance, toric matrix, toric code, dualcode, generator]`, where `vertices in first order` means ordering vertices by the coordinates from lowest to highest in each coordinate,  `vertices in second order` means ordering vertices by their distance from the origin, `toric matrix` means the initial matrix one can generate for the code using vertices.

- `dataset_10e5_F7_by_n_vert` contains files of the form `dataset_100000_n_F7.txt.zip`, where `n` is the number between 5 and 31 (inclusive). Each file is similar to the file in `dataset_10e5_F7_initial` folder, but now the features are: `[vertices, minimum hamming distance of toric code, generator, dual generator]`

For $\mathbb{F}_7$ codes, there are three datasets
- `dataset_F8_10e5` with 100,000 codes for number of vertices $4 \dots 17$ (inclusive),
- `dataset_F8_10e5` with 1,000 codes for number of vertices $16 \dots 45$ (inclusive),
- `dataset_F8_aux` with  up to 20,000 codes (if so many of them exists, otherwise less) for number of vertices $2,3,40 \dots 47$ (inclusive).
The features in all the three files are:
 `[minimum hamming distance of toric code, first order approximation of hamming distance, generator, dual generator]`
Recall that there is no primitive element as a number for $\mathbb{F}_8$, so matrices for $\mathbb{F}_8$ should be written in terms of $\alpha$, which denotes a primitive element for $\mathbb{F}_8$. In the files, $\alpha$ is represented as `$.1`.

## File_parsing

Contains scripts used parse text files and save them as pickle files which can be further loaded into torch for further processing or training.

For $\mathbb{F}_8$ codes files, we casted `$.1` powers into integers using the following rule:
$0 \rightarrow 0, 1 \rightarrow 1, $\alpha$ \rightarrow 2, $\alpha$^2 \rightarrow 3, $\alpha$^3 \rightarrow 4, $\alpha$^4 \rightarrow 5, $\alpha$^5 \rightarrow 6, $\alpha$^6 \rightarrow 7$.

## Model_training

Contains two scripts `transformer_3_F_7.py` and `transformer_436_vector_hpc_F_8.py` used to train the models employed in the Genetic Algorith search for  $\mathbb{F}_7$ and  $\mathbb{F}_8$ codes, respectively.

There are alsl two folders with checkpoints for the above models which can be loaded using `load` method of the GPT2mindist/ToricTransformer class (`optimizer` argument can be `none`).

## F7_GA

Contains the necedssary functions for running Genetic Algorithm and the main `ToricGA` python script/notebook to run the Genetic algorithm search.

## F7_Runs

Contains the results of the GA runs