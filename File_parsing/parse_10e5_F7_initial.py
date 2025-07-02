import re
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)
global_dtype = torch.float64
global_np_dtype = np.float64


def parse_matrix_block(block):
    # Remove brackets and split into rows
    block = block.strip().replace('[', '').replace(']', ',')[:-1]
    rows = block.split(',')
    matrix = []
    for row in rows:
        # Extract integers from each row
        matrix.append(list(map(int, row.split())))

    return np.array(matrix)


def parse_matrix_block2(block):
    # Remove brackets and split into rows
    block = block.strip().replace('(', '').replace(')', '')
    rows = block.split(',')
    matrix = []
    for row in rows:
        # Extract integers from each row
        matrix.append(list(map(int, row.split())))

    return np.array(matrix)


def parse_code_information(file_content):
    # Regex patterns to extract data
    coord_pattern = re.compile(r'<(\d+),\s*(\d+)>')
    code_info_pattern = re.compile(r'\[\s*(\d+),\s*(\d+),\s*(\d+)\]\s*Linear Code over GF\((\d+)\)')
    code_info_pattern_2 = re.compile(
        r'\[\s*(\d+),\s*(\d+),\s*(\d+)\]\s*Quasicyclic of degree 6 Linear Code over GF\((\d+)\)')
    code_info_pattern_3 = re.compile(r'\[\s*(\d+),\s*(\d+),\s*(\d+)\]\s*Cyclic Linear Code over GF\((\d+)\)')

    blocks = file_content[4:-5].split('*], [*')

    parsed_data = []

    for block in blocks:

        subblocks = block.split(';')

        # Find coordinates (pairs inside <>)
        coordinates1 = coord_pattern.findall(subblocks[0])
        coordinates1 = [(int(x), int(y)) for x, y in coordinates1]

        # Find coordinates (pairs inside <>)
        coordinates2 = coord_pattern.findall(subblocks[1])
        coordinates2 = [(int(x), int(y)) for x, y in coordinates2]

        mindist = int(subblocks[2])
        mindist_approx = int(subblocks[3])

        precode = parse_matrix_block(subblocks[4])

        ## Parsing generator subblock
        # Find code info (dimensions and field size)
        code_info_match = code_info_pattern.search(subblocks[5])
        code_info_match_2 = code_info_pattern_2.search(subblocks[5])
        code_info_match_3 = code_info_pattern_3.search(subblocks[5])

        if code_info_match:
            code_info1 = list(map(int, code_info_match.groups()))
        elif code_info_match_2:
            code_info1 = list(map(int, code_info_match_2.groups()))
        elif code_info_match_3:
            code_info1 = list(map(int, code_info_match_3.groups()))
        else:
            code_info1 = None

        # Find and parse the generator matrix
        matrix_start = subblocks[5].find('Generator matrix:') + len('Generator matrix:')
        matrix_block = subblocks[5][matrix_start:]
        generator = parse_matrix_block(matrix_block)

        ## Parsing dual generator subblock
        # Find code info (dimensions and field size)
        code_info_match = code_info_pattern.search(subblocks[6])
        code_info_match_2 = code_info_pattern_2.search(subblocks[6])
        code_info_match_3 = code_info_pattern_3.search(subblocks[6])

        if code_info_match:
            code_info2 = list(map(int, code_info_match.groups()))
        elif code_info_match_2:
            code_info2 = list(map(int, code_info_match_2.groups()))
        elif code_info_match_3:
            code_info2 = list(map(int, code_info_match_3.groups()))
        else:
            code_info2 = None

        # Find and parse the generator matrix
        matrix_start = subblocks[6].find('Generator matrix:') + len('Generator matrix:')
        matrix_block = subblocks[6][matrix_start:]
        generator_dual = parse_matrix_block(matrix_block)

        code_bock = subblocks[7]
        code_bock = code_bock.strip().replace('{', '').replace('}', '')
        code2 = parse_matrix_block2(code_bock)

        parsed_data.append({
            'mindist': mindist,
            'mindist_approx': mindist_approx,
            'coordinates1': coordinates1,
            'coordinates2': coordinates2,
            'code_info1': code_info1,
            'generator': generator,
            'code_info2': code_info2,
            'generator_dual': generator_dual,
            'precode1': precode,
            'code2': code2
        })

    return parsed_data


def parse_file(filename):
    with open(filename, 'r') as file:
        file_content = file.read()
    return parse_code_information(file_content)


filename = 'dataset_10e5_F7_mod.txt'
parsed_array = parse_file(filename)

# Print the parsed data
for idx, block in enumerate(parsed_array):
    if block['code_info1'] is None:
        print(idx)
        print(f"Coordinates1: {block['coordinates1']}")
        print(f"Coordinates2: {block['coordinates2']}")
        print(f"Code Info1: {block['code_info1']}")
        print("Generator:")
        print(block['generator'])
        print("Generator dual:")
        print(block['generator_dual'])
        print("precode1:")
        print(block['precode1'])
        print("Code2:")
        print(block['code2'])
    if len(block['coordinates1']) < 2:
        print(idx)
    if block['mindist'] > 30:
        print(idx)
        print(block['coordinates1'])


data_len = len(parsed_array)
print(data_len)

preCode1_dataset = []
Code2_dataset = []
Codeinfo1_dataset = []
Codeinfo2_dataset = []
Generator_dataset = []
GeneratorDual_dataset = []
Coords1_dataset = []
Coords2_dataset = []
Mindist_dataset = []
MindistApprox_dataset = []

for block in parsed_array:
    preCode1_dataset.append(torch.from_numpy(block['precode1'].astype(global_np_dtype)))
    Code2_dataset.append(torch.from_numpy(block['code2'].astype(global_np_dtype)))
    Coords1_dataset.append(torch.tensor(block['coordinates1'], dtype=global_dtype))
    Coords2_dataset.append(torch.tensor(block['coordinates2'], dtype=global_dtype))
    Mindist_dataset.append(torch.tensor(block['mindist'], dtype=global_dtype))
    MindistApprox_dataset.append(torch.tensor(block['mindist_approx'], dtype=global_dtype))
    Codeinfo1_dataset.append(block['code_info1'])
    Generator_dataset.append(torch.from_numpy(block['generator'].astype(global_np_dtype)))
    Codeinfo1_dataset.append(block['code_info2'])
    GeneratorDual_dataset.append(torch.from_numpy(block['generator_dual'].astype(global_np_dtype)))


torch.save(preCode1_dataset, 'CodesF7_data_10e5/preCode1_dataset.pt')
torch.save(Code2_dataset, 'CodesF7_data_10e5/Code2_dataset.pt')
torch.save(Coords1_dataset, 'CodesF7_data_10e5/Coords1_dataset.pt')
torch.save(Coords2_dataset, 'CodesF7_data_10e5/Coords2_dataset.pt')
torch.save(Mindist_dataset, 'CodesF7_data_10e5/Mindist_dataset.pt')
torch.save(MindistApprox_dataset, 'CodesF7_data_10e5/MindistApprox_dataset.pt')
torch.save(Codeinfo1_dataset, 'CodesF7_data_10e5/Codeinfo1_dataset.pt')
torch.save(Generator_dataset, 'CodesF7_data_10e5/Generator_dataset.pt')
torch.save(Codeinfo2_dataset, 'CodesF7_data_10e5/Codeinfo2_dataset.pt')
torch.save(GeneratorDual_dataset, 'CodesF7_data_10e5/GeneratorDual_dataset.pt')

mindist_hist = []
for mindist in Mindist_dataset:
    mindist_hist.append(mindist)
print(min(mindist_hist))
print(max(mindist_hist))

plt.hist(mindist_hist, bins=100)
plt.yscale("log")
plt.show()
