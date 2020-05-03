'''
This file contains data reading and 
preprocessing code.

Running this file as a script will
read in data from a .csv file and
perform the necessary steps to 
use the RNA data with RNNG (model
not included)
'''

import pandas as pd

def read_data(filepath='../purged_RNA_secondary_structure.csv', chunksize=100, return_iter=False):
    '''
    Reads data from provided CSV file.

    Will only read the label, actual sequence, and dot-bracket
    into the DataFrame.

    If `return_iter` == True, returns iterator with `chunksize` elements per iteration

    Else, will only return the first chunk from the data.
    '''
    data = pd.read_csv(filepath, usecols=[0, 1, 2], names=['name', 'seq', 'struct'], index_col=0, chunksize=chunksize)
    # data.loc[:, 'seq'].str.lower()
    if return_iter: return data
    else: return data.get_chunk()

def convert_all_notation_rnng(chunksize=1000):
    '''
    Handles reading and converting entire CSV file to necesary notation.
    '''
    data = read_data(chunksize=chunksize, return_iter=True)
    
    parse_seqs = []
    parse_inds = []
    for data_chunk in data:
        seqs = data_chunk.loc[:, 'seq'].str.lower()
        structs = data_chunk.loc[:, 'struct']

        valid_seqs = []
        valid_structs = []
        valid_inds = []
        for i, r in enumerate(seqs):
            if validate_rna_sequence(r): 
                valid_seqs.append(r)
                valid_structs.append(structs[i])
                valid_inds.append(data_chunk.index[i])

        parse_seqs += [convert_notation_rnng(seq, struct) for seq, struct in zip(valid_seqs, valid_structs)]
        parse_inds += list(valid_inds)
    
    return pd.Series(parse_seqs, index=parse_inds)
        
    
def convert_notation_rnng(sequence, structure):
    '''
    `sequence` must be a string that contains an RNA sequence

    `structure` must be a string that contains the dot-bracket notation
    for the RNA sequence provided in `sequence`

    This function returns a notation combining the sequence and structure
    that mimics the input notation for RNNG (Dyer et al.).
    '''
    parse_seq = ['(S ']
    for n, s in zip(sequence, structure):
        if s == '.': parse_seq.append('(E {}) '.format(n))
        elif s == '(': parse_seq.append('(S (E {}) '.format(n))
        elif s == ')': parse_seq.append('(E {})) '.format(n))
        elif s == '<': parse_seq.append('(S (E {}) '.format(n))
        elif s == '>': parse_seq.append('(E {})) '.format(n))
        elif s == '{': parse_seq.append('(S (E {}) '.format(n))
        elif s == '}': parse_seq.append('(E {})) '.format(n))
        elif s == '[': parse_seq.append('(S (E {}) '.format(n))
        elif s == ']': parse_seq.append('(E {})) '.format(n))       
        elif s.isalpha() and s.isupper(): parse_seq.append('(S (E {}) '.format(n))
        elif s.isalpha() and s.islower(): parse_seq.append('(E {})) '.format(n))       
        else: raise AttributeError('Unknown structure token: {}'.format(s))

        print(n, s)
        print(parse_seq[-1])
    parse_seq.append(')')
    parse_seq_str = ''.join(parse_seq)
    if parse_seq_str.count('(') != parse_seq_str.count(')'):
        print(sequence)
        print(structure)
        print(parse_seq_str)
    return parse_seq_str


def write_data(data, dirpath='./'):
    '''
    Splits data into training, validation, and testing
    sets before writing each set to a separate txt files
    '''
    data_len = data.shape[0]
    train_len = int(data_len * 0.8)
    dev_len = int(data_len * 0.1)

    train_data = data.iloc[:train_len]
    dev_data = data.iloc[train_len:train_len + dev_len]
    test_data = data.iloc[train_len + dev_len:]

    train_data.to_csv(dirpath+'train_data.txt', sep='\n', index=False)
    dev_data.to_csv(dirpath+'dev_data.txt', sep='\n', index=False)
    test_data.to_csv(dirpath+'test_data.txt', sep='\n', index=False)

    print(train_data.shape)
    print(dev_data.shape)
    print(test_data.shape)

    print()
    print(train_data)
    print(dev_data)
    print(test_data)

def validate_rna_sequence(rna_seq):
    '''
    Validates the given RNA sequence only contains
    the four possible nucleotides: A, C, G, and U
    '''
    unique_chars = set(rna_seq)
    if unique_chars - {'a', 'c', 'g', 'u'} != set():
        return False
    else: return True
    

if __name__ == '__main__':
    proc_data = convert_all_notation_rnng()
    write_data(proc_data)

    print(proc_data.head())

    
