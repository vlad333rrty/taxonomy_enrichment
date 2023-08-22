def write_result(data_list, dest_path):
    """
    format: withdef.1	primer#n#1	attach - separator == \t
    :return:
    """
    contents = ''
    for x in data_list:
        contents += '{}\t{}\t{}\n'.format(x[0], '' if x[1] is None else x[1].name(), x[2])
    with open(dest_path, 'w') as file:
        file.write(contents)


def write_data_for_doc(data_list, path):
    """
    format word to add, first word from gloss, sysnsets of this word
    """
    contents = ''
    for data in data_list:
        contents += '{}, {}, {}\n'.format(data[0], data[1], data[2])
    with open(path, 'w') as file:
        file.write(contents)