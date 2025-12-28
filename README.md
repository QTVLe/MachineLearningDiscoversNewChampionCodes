# MachineLearningDiscoversChampionCodes
The abstract of the paper, _Machine Learning Discovers Champion Codes_:

Linear error-correcting codes form the mathematical backbone of modern digital communication and storage systems, but identifying champion linear codes (linear codes achieving or exceeding the best known minimum Hamming distance) remains challenging. By training a transformer to predict the minimum Hamming distance of a class of linear codes and pairing it with a genetic algorithm over the search space, we develop a novel method for discovering champion codes. This model effectively reduces the search space of linear codes needed to achieve a best minimum Hamming distance. Our results present the use of this method in the study and construction of error-correcting codes, applicable to codes such as generalised toric, Reed--Muller, Bose--Chaudhuri--Hocquenghem, Algebraic--Geometry, and potentially quantum codes.

<!-- This repository includes all runs exploring $\mathbb{F}_7$, seen in the $\mathbb{F}_7$ folder, and all runs exploring $\mathbb{F}_8$, seen in the $\mathbb{F}_8$ folder. Both also include all code relating to the project. -->

# Structure

## Magma code

Contains an example Magma CAS script to generate $\mathbb{F}_8$ dataset. Changing line 3 from `prime := 8;` to `prime := 7;` will generate $\mathbb{F}_7$ codes. To change the generated dataset size, alter a number in line 50. Number of vertices, for which the data is generated, can be changed in line 80.

## Datasets

Contains $\mathbb{F}_7$ and $\mathbb{F}_8$ datasets as text files as they were generated in Magma. Folders with `F_7` in the name contain datasets for $\mathbb{F}_7$ codes, with `F_8` - for $\mathbb{F}_8$. 

As explained in the paper, we generated two datasets for $\mathbb{F}_7$, initial one with 100,000 codes for all vertices - folder `dataset_10e5_F7_initial`, and the main one - `dataset_10e5_F7_by_n_vert`, with 100,000 codes for each generated number of vertices.

- `dataset_10e5_F7_initial` contains file `dataset_10e5_F7.txt.zip`, which is orgainised as a list, where each element is a list with the following features: `[vertices in first order, vertices in second order, minimum hamming distance of toric code, first order approximation of hamming distance, toric matrix, toric code, dualcode, generator]`, where `vertices in first order` means ordering vertices by the coordinates from lowest to highest in each coordinate,  `vertices in second order` means ordering vertices by their distance from the origin, `toric matrix` means the initial matrix one can generate for the code using vertices.

- `dataset_10e5_F7_by_n_vert` contains files of the form `dataset_100000_n_F7.txt.zip`, where `n` is the number between 5 and 31 (inclusive). Each file is similar to the file in `dataset_10e5_F7_initial` folder, but now the features are: `[vertices, minimum hamming distance of toric code, generator, dual generator]`

For $\mathbb{F}_8$ codes, there are three datasets
- `dataset_F8_10e5` with 100,000 codes for number of vertices $4 \dots 17$ (inclusive),
- `dataset_F8_10e5` with 1,000 codes for number of vertices $16 \dots 45$ (inclusive),
- `dataset_F8_aux` with  up to 20,000 codes (if so many of them exists, otherwise less) for number of vertices $2,3,40 \dots 47$ (inclusive).
The features in all the three files are:
 `[minimum hamming distance of toric code, first order approximation of hamming distance, generator, dual generator]`
Recall that there is no primitive element as a number for $\mathbb{F}_8$, so matrices for $\mathbb{F}_8$ should be written in terms of $\alpha$, which denotes a primitive element for $\mathbb{F}_8$. In the files, $\alpha$ is represented as `$.1`.

## File_parsing

Contains scripts used parse text files and save them as pickle files which can be further loaded into torch for further processing or training.

For $\mathbb{F}_8$ codes files, we casted `$.1` powers into integers using the following rule:
$0 \rightarrow 0, 1 \rightarrow 1, \alpha \rightarrow 2, \alpha^2 \rightarrow 3, \alpha^3 \rightarrow 4, \alpha^4 \rightarrow 5, \alpha^5 \rightarrow 6, \alpha^6 \rightarrow 7$.

## Model_training

Contains two scripts `transformer_3_F_7.py` and `transformer_436_vector_hpc_F_8.py` used to train the models employed in the Genetic Algorith search for  $\mathbb{F}_7$ and  $\mathbb{F}_8$ codes, respectively.

There are also two folders with checkpoints for the above models which can be loaded using `load` method of the GPT2mindist/ToricTransformer class (`optimizer` argument can be `none`).

## F_7_GA

Contains the necedssary functions for running Genetic Algorithm and the main `ToricGA` python script/notebook to run the Genetic algorithm search.

Similar for `F_8_GA` folder.

## F_7_Runs

Contains the results of the GA runs.

Similar for `F_8_Runs` folder.
