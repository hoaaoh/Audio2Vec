from utils import *

feat_dir = '/nfs/YueLao/grtzsohalf/yeeee/English'
seq_len = 70
n_files = 200
proportion = 0.975

def main():
    split_data(feat_dir, n_files, proportion, seq_len)

if __name__=='__main__':
    main()
