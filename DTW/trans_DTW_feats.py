#!/usr/bin/env python3

import argparse
import gen_len as GL

FLAG = None 

def write_DTW_feats(feats, labs, len_list):
    with open(FLAG.out_scp,'a') as f_scp:
        for i in range(len(labs)):
            feat_len = len_list[i]
            lab = labs[i]
            file_name= FLAG.out_dir+'/'+str(FLAG.counter)+'_'+str(int(lab))+'.feat'
            print (file_name)
            with open(file_name,'w') as f:
                f.write('LT='+str(feat_len)+',LF='+str(FLAG.feat_dim)+',dT=0.01\n')
                for j in range(feat_len):
                    for k in range(FLAG.feat_dim-1):
                        f.write(str(feats[i][j*FLAG.feat_dim+k]) + ' ')
                    f.write(str(feats[i][j*FLAG.feat_dim+FLAG.feat_dim-1])+'\n')

            FLAG.counter += 1
            f_scp.write(file_name+'\n')        

    return 

def main():
    feats, labs = GL.read_csv(FLAG.ark_file)
    len_list = GL.gen_len(feats)
    write_DTW_feats(feats, labs, len_list)
    return

if __name__=='__main__':
    parser=argparse.ArgumentParser(
        description='generate the dtw needed features')
    parser.add_argument('--feat_dim', type=int, default=39,
        metavar='<feature dimension>')
    parser.add_argument('ark_file',
        help='the transforming ark file')
    parser.add_argument('out_dir',
        help='the output directory')
    parser.add_argument('out_scp',
        help='the scp of the output files')
    parser.add_argument('counter', type=int,
        help='the output file counter')
    FLAG = parser.parse_args()
    GL.FLAG = FLAG
    main()
