import os

# src='output/wavelet_tiny_patch4_224/default/log_rank0.txt'
# dst='acc.txt'

# with open(src, 'r') as f:
    # lines = f.readlines()
    
# fid = open(dst, 'w')
# for line in lines:
    # if 'INFO Max accuracy' in line:
        # fid.write(line.split(':')[-1])
# fid.close()    

src='ckpt/log.txt'
dst='acc.txt'
with open(src, 'r') as f:
    lines = f.readlines()
    
fid = open(dst, 'w')
for line in lines:
    a = line.split(':')[6]
    a = a.split(',')[0].strip()+'%\n'   
    fid.write(a)
fid.close()      
    