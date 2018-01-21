from utils import *

feat_dir = '/home_local/grtzsohalf/yeeee/German'
seq_len = 70
n_files = 8166
proportion = 0.

def main():
    split_data(feat_dir, n_files, proportion, seq_len)

if __name__=='__main__':
    main()
