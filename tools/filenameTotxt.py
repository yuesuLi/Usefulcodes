import os
# dir="/media/ourDataset/v1.0/20211025_1_old/Dataset/group0000_frame0003"
dir="/media/ourDataset/v1.0_label"

f=open("dir.txt","w")
for root,dirs,files in os.walk(dir):
    # files.sort(key=lambda x:(int(x[-14:-9])*1000 + int(x[-7:-4])))
    for file in files:
        if file[-3:] == 'png':
            # tmp = os.path.join(root,file)
            # print(int(tmp[-14:-8])*1000)
            # print(tmp[-7:-4])

            f_str = str(os.path.join(root, file))
            print(len(f_str))
            # print(os.path.join(root,file))
            # f.writelines(os.path.join(root,file)+"\n")
