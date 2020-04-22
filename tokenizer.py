def tokenizer(sequence, structure):
    """
    Tokenizer will parse the input sequence based on the
    secondary structure of the RNA sequence
    Args:
        sequence: input sequence
        structure: the secondary structure of the RNA sequence

    Returns:
        left_paren_bucket: sequences with '(' secondary structure
        right_paren_bucket: sequences with ')' secondary structure
        less_than_bucket: sequences with '<' secondary structure
        greater_than_bucket: sequences with '>' secondary structure
        dot_bucket: sequences with '.' secondary structure
    """
    # make sure the dimensions match
    if len(sequence) != len(structure):
        print("the input sequence and the secondary structure have different dimensions")
        return
    # group the sequence with the same dot-parentheses representations together
    left_paren_bucket = list()
    right_paren_bucket = list()
    less_than_bucket = list()
    greater_than_bucket = list()
    dot_bucket = list()

    group_start = 0
    for i in range(len(structure)):
        if structure[i] != structure[group_start]:
            if structure[group_start] == '(':
                left_paren_bucket.append(sequence[group_start: i])
            elif structure[group_start] == ')':
                right_paren_bucket.append(sequence[group_start: i])
            elif structure[group_start] == '<':
                less_than_bucket.append(sequence[group_start: i])
            elif structure[group_start] == '>':
                greater_than_bucket.append(sequence[group_start: i])
            else:
                dot_bucket.append(sequence[group_start: i])
            group_start = i

    return left_paren_bucket, right_paren_bucket, less_than_bucket, greater_than_bucket, dot_buckettokenizer.py