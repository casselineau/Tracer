import numpy as N

def export_hist_data_csv(hist, bins, saveloc, info_header, separator=','):
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
        fo.write('bins_x' + separator)
        for e in bins_x:
            fo.write(str(e) + separator)
        fo.write('\n')
        if dims == 2: # if it is a 2 D histogram
            fo.write('bins_y' + separator)
            for e in bins_y:
                fo.write(str(e) + separator)
            fo.write('\n')
            for l in range(hist.shape[0]):
                for f in hist[l]:
                    fo.write(str(f) + ',')
                fo.write('\n')
        else:
            for f in hist:
                fo.write(str(f) + ',')

def load_hist_data_csv(fluxmap_file):
    with open(fluxmap_file, 'r') as fo:
        data = fo.readlines()

    print('Data information:\n', data[0]) # Info header

    bins = []
    for b in data[1:]:
        if 'bins_' in b:
            bins.append(N.array(b.split(',')[1:-1], dtype=float))
    if len(bins) == 1:
        load_data = N.array(data[2].split(',')[:-1], dtype=float)
    if len(bins) == 2:
        load_data = N.zeros((len(bins[0])-1, len(bins[1])-1))
        for d in range(load_data.shape[0]):
            load_data[d] = N.array(data[3 + d].split(',')[:-1], dtype=float)
    return bins, load_data