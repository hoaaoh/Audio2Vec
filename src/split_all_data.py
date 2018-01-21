import os

feat_dir = '/home_local/grtzsohalf/yeeee/English'
seq_len = 70
feats_dir = os.path.join(feat_dir, 'feats', str(seq_len))
n_spks = len(os.listdir(feats_dir))
print (n_spks)
n_spks_per_file = 200
n_files = n_spks // n_spks_per_file

def main():
    for i in range(n_files+1):
        test_scp = os.path.join(feat_dir, 'all_AE/all_AE_'+str(i)+'.scp')
        if i != n_files:
            file_list = os.listdir(feats_dir)[n_spks_per_file*i:n_spks_per_file*(i+1)]
        if i == n_files:
            file_list = os.listdir(feats_dir)[n_spks_per_file*(i):]
        # random.shuffle(file_list)
        print ("Testing number of speakers in list "+str(i)+": " + str(len(file_list)))
        with open(test_scp, 'w') as fout:
            for f in file_list:
                fout.write(f + '\n')

if __name__=='__main__':
    main()
