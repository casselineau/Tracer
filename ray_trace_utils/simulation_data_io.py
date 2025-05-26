import numpy as N

def save_hist_data_csv(hist, bins, hist_label, bins_label, info_header, saveloc, separator=','):
    '''
    Exports a fluxmap in .csv format
    :param hist:
    :param bins:
    :return:
    '''
    dims = hist.ndim
    if dims == 1:
        bins_x = bins
    if dims == 2:
        bins_x = bins[0]
        bins_y = bins[1]

    # Write fluxmap to file
    with open(saveloc, 'w') as fo:
        fo.write(info_header)
        fo.write('\n')
        if dims == 2:
            fo.write('bins_x:'+separator+bins_label[0])
        else:
            fo.write('bins_x:'+separator+bins_label)
        fo.write('\n')
        for e in bins_x:
            fo.write(str(e) + separator)
        fo.write('\n')
        if dims == 2: # if it is a 2 D histogram
            fo.write('bins_y:'+separator+bins_label[1])
            fo.write('\n')
            for e in bins_y:
                fo.write(str(e) + separator)
            fo.write('\n')
            fo.write('data:'+separator+hist_label)
            fo.write('\n')
            for l in range(hist.shape[0]):
                for f in hist[l]:
                    fo.write(str(f) + separator)
                fo.write('\n')
        else:
            fo.write('data:'+separator+hist_label)
            fo.write('\n')
            for f in hist:
                fo.write(str(f) + separator)

def load_hist_data_csv(fluxmap_file, separator=','):
    with open(fluxmap_file, 'r') as fo:
        data = fo.readlines()

    print('Data information:\n', data[0]) # Info header

    bins = []
    bins_label = []
    load_data = []
    for i, b in enumerate(data[1:]):
        if 'bins_' in b:
            bins_label.append(b.split(separator)[-1])
            bins.append(N.array(data[i+2].split(separator)[:-1], dtype=float))
        if 'data' in b:
            data_label = b.split(separator)[-1]
            if len(bins) == 1:
                load_data = N.array(data[i+2].split(separator)[:-1], dtype=float)
            if len(bins) == 2:
                for j in range(len(bins[0])-1):
                    load_data.append([d for d in N.array(data[i+2+j].split(separator)[:-1], dtype=float)])
    if len(bins) == 1:
        bins = bins[0]
        bins_label = bins_label[0]
    else:
        load_data = N.array(load_data)
    return bins, load_data, bins_label, data_label