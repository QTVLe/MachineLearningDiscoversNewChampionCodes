# gigaDataset_F8

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

torch.set_default_dtype(torch.float32)
global_dtype = torch.float32
global_np_dtype = np.float32

symbol_mapping = {'0':0, '1':1, '$.1':2, '$.1^2':3, '$.1^3':4, '$.1^4':5, '$.1^5':6, '$.1^6':7}

def parse_file(filename, num_subblocks=3):
    with open(filename, 'r') as file:
        file_content = file.read()

        blocks = file_content[7:-4].split('],\n    [')
        parsed_data = []

        for block in blocks:
            block_data = parse_block(block, num_subblocks=num_subblocks)
            if block_data is not None:
                parsed_data.append(block_data)

    return parsed_data

def parse_block(block, num_subblocks = 3):
    subblocks = block.strip().split(';')

    mindist = int(subblocks[0])
    if mindist == -1:
        return None

    if num_subblocks == 3:
        mindist_approx = int(subblocks[1])
        matrix = parse_matrix(subblocks[2])
        parsed_data = {
                'generator_matrix': np.array(matrix, dtype=np.int8),
                'mindist': mindist,
                'mindist_approx': mindist_approx
            }
        return parsed_data
    elif num_subblocks == 2:
        matrix = parse_matrix(subblocks[1])
        parsed_data = {
                'generator_matrix': np.array(matrix, dtype=np.int8),
                'mindist': mindist
            }
        return parsed_data
    else:
        ValueError("Number of blocks num_subblocks is not 2 or 3!")

def parse_matrix(data):
    rows = data.replace('{','').replace('}','').split('),\n    (')
    for i in range(len(rows)):
        print(rows[i])
        rows[i] = rows[i].replace('\n    ', ' ').replace('(','').replace(')','').replace('  ',' ').strip().split(' ')
        print(rows[i])
        rows[i] = list(map(symbol_mapping.get, rows[i]))
        print(rows[i])
        break
    return rows
 
# parsing codesF8_10e5_textdata
for n in range(4, 17+1):
    parsed_data = parse_file(f"C:\\Users\\adcs653\\OneDrive - City, University of London\\ML Toric codes\\codesF8_10e5_textdata\\dataset_100000_{n}_F8.txt")

    GeneratorDataset = []
    MindistDataset = []
    MindistApproxDataset = []
    for block in parsed_data:
        MindistDataset.append(torch.tensor(block['mindist'], dtype=torch.int8))
        MindistApproxDataset.append(torch.tensor(block['mindist_approx'], dtype=torch.int8))
        GeneratorDataset.append(torch.from_numpy(block['generator_matrix']))

    directory = f"codesF8_10e5_by_mindist\\codes_{n}F8_10e5\\"
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)

    torch.save(MindistDataset, directory+"MindistDataset.pt")
    torch.save(MindistApproxDataset, directory+"MindistApproxDataset.pt")
    torch.save(GeneratorDataset, directory+"GeneratorDataset.pt")

    print(f"dataset {n} length: ", len(MindistDataset))

# # parsing codesF8_10e3_textdata
# for n in range(16, 45+1):
#     parsed_data = parse_file(f"C:\\Users\\adcs653\\OneDrive - City, University of London\\ML Toric codes\\codesF8_10e3_textdata\\dataset_{n}_F8.txt", num_subblocks=2)

#     GeneratorDataset = []
#     MindistDataset = []
#     uncomputed_counter = 0
#     for block in parsed_data:
#         if block['mindist'] == -1:
#             uncomputed_counter += 1
#         MindistDataset.append(torch.tensor(block['mindist'], dtype=torch.int8))
#         GeneratorDataset.append(torch.from_numpy(block['generator_matrix']))

#     directory = f"codesF8_10e3_by_mindist\\codes_{n}F8_10e3\\"
#     p = Path(directory)
#     p.mkdir(parents=True, exist_ok=True)

#     torch.save(MindistDataset, directory + "MindistDataset.pt")
#     torch.save(GeneratorDataset, directory + "GeneratorDataset.pt")

#     print(f"dataset {n} length: ", len(MindistDataset))
#     print(f"Number of uncomputed mindists: {uncomputed_counter}")

# # parsing codesF8_aux_textdata
# codes_range = [2,3,40,41,41,43,44,45,46,47]
# for n in codes_range:
#     parsed_data = parse_file(f"C:\\Users\\adcs653\\OneDrive - City, University of London\\ML Toric codes\\codesF8_aux_textdata\\datasetnew_{n}_F8.txt", num_subblocks=2)

#     GeneratorDataset = []
#     MindistDataset = []
#     uncomputed_counter = 0
#     for block in parsed_data:
#         if block['mindist'] == -1:
#             uncomputed_counter += 1
#         MindistDataset.append(torch.tensor(block['mindist'], dtype=torch.int8))
#         GeneratorDataset.append(torch.from_numpy(block['generator_matrix']))

#     directory = f"codesF8_aux_by_mindist\\codes_{n}F8\\"
#     p = Path(directory)
#     p.mkdir(parents=True, exist_ok=True)

#     torch.save(MindistDataset, directory + "MindistDataset.pt")
#     torch.save(GeneratorDataset, directory + "GeneratorDataset.pt")

#     print(f"dataset {n} length: ", len(MindistDataset))
#     print(f"Number of uncomputed mindists: {uncomputed_counter}")