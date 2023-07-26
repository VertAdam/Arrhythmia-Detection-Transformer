import wfdb

# Grabbing only the 10th sequence for the first 1000 patients
db = 'icentia11k-continuous-ecg'
record = ['p00/p0000%s/p0000%s_s10'%(n,n) for n in range(0,10)]+  \
         ['p00/p000%s/p000%s_s10'%(n,n) for n in range(10,100)]+\
         ['p00/p00%s/p00%s_s10'%(n,n) for n in range(100,1000)]

# download the record
wfdb.dl_database(db, dl_dir='icentia11k data', records=record)

x = 1