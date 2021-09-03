def column_align(texts):
    ret = list()
    for i, line in enumerate(texts):
        line = line[:-1]
        line = line.split('\t')

        if len(line) != 10:
            print(f'{i+1} ({len(line)}): {line}')
        ret.append('\t'.join(line))
    return ret


if __name__ == '__main__':
    file = './datasets/raw/traindata/video_features_data.csv'
    f = open(file, 'r')
    text = f.readlines()
    f.close()
    text = column_align(text)
    g = open(file, 'w')
    g.write('\n'.join(text))
    g.close()