def vec_load(file_name, file_path, num):
    """
    load vec dict file
    :param num: 读取字典中单词的数量
    :param file_name:
    :param file_path:
    :return: a dict of words
    """
    dct = {}
    f = open(file_path + '/' + file_name, 'r', encoding='utf-8')
    line = f.readline()
    pin = 0
    while line:
        try:
            line = line.split()
            dct[line[0]] = [float(val) for val in line[1:]]
            pin += 1
            if pin >= num:
                break
        except:
            print(line)
        line = f.readline()
    return dct


if __name__ == '__main__':
    file_name = "glove.6B.50d.txt"
    file_path = "D:/OneDrive/WORK/word_vector"
    dct = vec_load(file_name, file_path)
