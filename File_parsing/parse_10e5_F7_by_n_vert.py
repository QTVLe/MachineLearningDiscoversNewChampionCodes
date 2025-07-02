import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import zipfile

from pathlib import Path
# import argparse

from multiprocessing import Pool

# parser = argparse.ArgumentParser(description='Parse file with specified number of vertices')
# parser.add_argument('--n_vertices', action="store", dest='n_vertices')
# args = parser.parse_args()
# n_vertices = vars(args)['n_vertices']

torch.set_default_dtype(torch.float32)
global_dtype = torch.float32
global_np_dtype = np.float32

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


def parse_code_block(block):
    # Regex patterns to extract data
    coord_pattern = re.compile(r'<(\d+),\s*(\d+)>')
    code_info_pattern = re.compile(r'\[\s*(\d+),\s*(\d+),\s*(\d+)\]\s*Linear Code over GF\((\d+)\)')
    code_info_pattern_2 = re.compile(
        r'\[\s*(\d+),\s*(\d+),\s*(\d+)\]\s*Quasicyclic of degree 6 Linear Code over GF\((\d+)\)')
    code_info_pattern_3 = re.compile(r'\[\s*(\d+),\s*(\d+),\s*(\d+)\]\s*Cyclic Linear Code over GF\((\d+)\)')

    subblocks = block.split(';')

    # Find coordinates (pairs inside <>)
    coordinates = coord_pattern.findall(subblocks[0])
    coordinates = [(int(x), int(y)) for x, y in coordinates]

    mindist = int(subblocks[1])

    ## Parsing generator subblock
    # Find code info (dimensions and field size)
    code_info_match = code_info_pattern.search(subblocks[2])
    code_info_match_2 = code_info_pattern_2.search(subblocks[2])
    code_info_match_3 = code_info_pattern_3.search(subblocks[2])

    if code_info_match:
        code_info1 = list(map(int, code_info_match.groups()))
    elif code_info_match_2:
        code_info1 = list(map(int, code_info_match_2.groups()))
    elif code_info_match_3:
        code_info1 = list(map(int, code_info_match_3.groups()))
    else:
        code_info1 = None

    # Find and parse the generator matrix
    matrix_start = subblocks[2].find('Generator matrix:') + len('Generator matrix:')
    matrix_block = subblocks[2][matrix_start:]
    generator = parse_matrix_block(matrix_block)

    ## Parsing dual generator subblock
    # Find code info (dimensions and field size)
    code_info_match = code_info_pattern.search(subblocks[3])
    code_info_match_2 = code_info_pattern_2.search(subblocks[3])
    code_info_match_3 = code_info_pattern_3.search(subblocks[3])

    if code_info_match:
        code_info2 = list(map(int, code_info_match.groups()))
    elif code_info_match_2:
        code_info2 = list(map(int, code_info_match_2.groups()))
    elif code_info_match_3:
        code_info2 = list(map(int, code_info_match_3.groups()))
    else:
        code_info2 = None

    # Find and parse the generator matrix
    matrix_start = subblocks[3].find('Generator matrix:') + len('Generator matrix:')
    matrix_block = subblocks[3][matrix_start:]
    generator_dual = parse_matrix_block(matrix_block)

    code_bock = subblocks[4]
    code_bock = code_bock.strip().replace('{', '').replace('}', '')
    code = parse_matrix_block2(code_bock)

    parsed_data = {
        'mindist': mindist,
        'coordinates': coordinates,
        'code_info1': code_info1,
        'generator': generator,
        'code_info2': code_info2,
        'generator_dual': generator_dual,
        'code': code
    }

    return parsed_data

def parse_code_information(file):
    parsed_data = []
    block_buffer = ""
    search_start = 0

    pattern = r"\*\]\s*,\s*\[\*"

    for idx, line in enumerate(file):
        block_buffer = block_buffer + line.decode("utf-8")  # collect lines in a block until we find the delimiter
        block_buffer = block_buffer.replace("\n", "")  # clean from any newline characters

        matches = list(re.finditer(pattern, block_buffer))

        if len(matches) > 0:
            # process the block

            # find begining and end of block
            start_pos = block_buffer.find("[*")
            end_pos = block_buffer.find("*]", start_pos)

            block = block_buffer[start_pos:end_pos].strip()

            parsed_data.append(parse_code_block(block))

            next_start = matches[-1].start()

            block_buffer = block_buffer[(next_start+2):]  # put the remainder to the buffer for further reading

    # parse very last block
    start_pos = block_buffer.find("[*")
    end_pos = block_buffer.find("*]", start_pos)

    block = block_buffer[start_pos:end_pos].strip()
    parsed_data.append(parse_code_block(block))
    
    return parsed_data


def parse_file(n_vertices):
    
    filename = "dataset_100000_" + str(n_vertices) + "_F7.txt"
    zipname = "C:\\Users\\Dimr7\\YandexDisk\\City PhD\\ML and NN\\Toric Codes\\Datasets\\F7_varSeqLen_10e5\\dataset_100000_" + str(n_vertices) + "_F7.txt.zip"
    
    with zipfile.ZipFile(zipname, 'r') as zf:
        with zf.open(filename) as file:
            parsed_array = parse_code_information(file) 

    # data_len = len(parsed_array)
    
    Code_dataset = []
    Codeinfo1_dataset = []
    Codeinfo2_dataset = []
    Generator_dataset = []
    GeneratorDual_dataset = []
    Coords_dataset = []
    Mindist_dataset = []

    for block in parsed_array:
        Code_dataset.append(torch.from_numpy(block['code'].astype(global_np_dtype)))
        Coords_dataset.append(torch.tensor(block['coordinates'], dtype=global_dtype))
        Mindist_dataset.append(torch.tensor(block['mindist'], dtype=global_dtype))
        Codeinfo1_dataset.append(block['code_info1'])
        Generator_dataset.append(torch.from_numpy(block['generator'].astype(global_np_dtype)))
        Codeinfo1_dataset.append(block['code_info2'])
        GeneratorDual_dataset.append(torch.from_numpy(block['generator_dual'].astype(global_np_dtype)))


    foldername = "Codes_" + str(n_vertices) + "F7_10e5/"
    Path(foldername).mkdir(parents=True, exist_ok=True)

    torch.save(Code_dataset,         foldername + 'Code2_dataset.pt')
    torch.save(Coords_dataset,       foldername + 'Coords1_dataset.pt')
    torch.save(Mindist_dataset,      foldername + 'Mindist_dataset.pt')
    torch.save(Codeinfo1_dataset,    foldername + 'Codeinfo1_dataset.pt')
    torch.save(Generator_dataset,    foldername + 'Generator_dataset.pt')
    torch.save(Codeinfo2_dataset,    foldername + 'Codeinfo2_dataset.pt')
    torch.save(GeneratorDual_dataset,foldername + 'GeneratorDual_dataset.pt')

    # mindist_hist = []
    # for mindist in Mindist_dataset:
    #     mindist_hist.append(mindist)
    # print(min(mindist_hist))
    # print(max(mindist_hist))

    # plt.hist(mindist_hist, bins=100)
    # plt.yscale("log")
    # plt.show()

if __name__ == '__main__':
        with Pool(processes=4) as p:
            p.map(parse_file, [17, 18, 19, 20]) 