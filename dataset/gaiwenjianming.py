import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

def gaiwenjianming():
    root = "/workspace/xwh/aster/train/all_images/"
    with open(os.path.join(root, "valid.txt"), 'w') as write_txt:
        with open(os.path.join(root, "valid_oral.txt"), 'r') as train_txt:
            lines = train_txt.readlines()
            for line in lines:
                cut = line.split('/', 2)
                write_txt.writelines(cut[2])

def open_picture():
    from PIL import Image
    import numpy as np
    img_PIL = Image.open("/workspace/xwh/aster/train/all_images/train_images/gt_24408_7.jpg")
    img_PIL.show()

def find_blank():
    root = "/workspace/xwh/aster/train/all_images/"
    with open(os.path.join(root, "train.txt"), 'r') as train_txt:
        lines = train_txt.readlines()
        for line in lines:
            print(line)
            cut = line.rstrip().split(' ', 1)
            print(cut[1])

def find_label_length():
    root = "/workspace/xwh/aster/train/all_images/"
    count = 0
    with open(os.path.join(root, "train.txt"), 'r') as train_txt:
        lines = train_txt.readlines()
        for line in lines:
            # print(line)
            cut = line.rstrip().split(' ', 1)
            # print(cut[0])
            # print(cut[1])
            try:
                cut[1]
            except:
                continue
            # print(len(cut[1]))
            if(len(cut[1]) > 60):
                count += 1
        print(count)

def delete_lines():
    root = "/workspace/xwh/aster/train/all_images/"
    count = 0
    with open(os.path.join(root, "tr.txt"), 'w') as write_txt:
        with open(os.path.join(root, "train.txt"), 'r') as train_txt:
            lines = train_txt.readlines()
            for line in lines:
                cut = line.rstrip().split(' ', 1)
                try:
                    cut[1]
                except:
                    continue
                # print(len(cut[1]))
                if (len(cut[1]) <= 40):
                    write_txt.writelines(line)

def txt_paixu():
    # root = "/workspace/datasets/mlt19/xwh/trainset/"
    root = "/workspace/xwh/aster/train/CASIA-10k/all/"
    count = 0
    with open(os.path.join(root, "alll.txt"), 'r') as read_txt:
        with open(os.path.join(root, "train_paixu.txt"), 'w') as write_txt:
            lines = read_txt.readlines()
            lines = sorted(lines)
            for line in lines:
                write_txt.writelines(line)

def tiqu_chinese():
    # root = "/workspace/datasets/mlt19/xwh/trainset/"
    root = "/workspace/datasets/ICDAR17/ICDAR17/z_grp_ccx/"
    count = 0
    with open(os.path.join(root, "train_paixu.txt"), 'r') as read_txt:
        with open(os.path.join(root, "all_chinese.txt"), 'w') as write_txt:
            lines = read_txt.readlines()
            for line in lines:
                gt = line.strip().split(",", 2)
                if gt[1] == "Chinese":
                    newline = gt[0] + " " + gt[2] + '\n'
                    write_txt.writelines(newline)

def copy_image():
    root = "/workspace/xwh/aster/train/CASIA-10k/all/"
    image_root = "/workspace/xwh/aster/train/CASIA-10k/all/image90/"
    count = 0
    print("1")
    with open(os.path.join(root, "train_paixu.txt"), 'r') as read_txt:
        lines = read_txt.readlines()
        for line in lines:
            count += 1
            print(count)
            gt = line.strip().split(" ")
            # print(gt[0])
            image = os.path.join(image_root, gt[0])
            shutil.copy(image, "/workspace/xwh/aster/train/all_images/train_images/")

def crop_image():
    root = "/workspace/xwh/aster/train/CASIA-10k/test/"
    to_image_root = "/workspace/xwh/aster/train/CASIA-10k/all/images/"
    to_txt_root = "/workspace/xwh/aster/train/CASIA-10k/all/"
    txt_write = open(os.path.join(to_txt_root, "test_casia.txt"), 'w')
    gts = os.listdir(root)
    gts = sorted(gts)
    # print(gts)
    count_file = 0
    for gt in gts:
        count = 0
        if gt.endswith('.txt'):
            count_file += 1
            print(count_file)
            # print("1", gt)
            image_name = gt.split('.')[0]+ ".jpg"
            imgSrc = Image.open(os.path.join(root, image_name))
            imgSrc = imgSrc.convert('RGB')
            try:
                with open(os.path.join(root, gt), 'r', encoding='gb18030') as rs:
                    rs = rs.readlines()
                    for r in rs:
                        ss = r.strip().split(',', 8)
                        if ss[8] != '###':
                            min_x = min(int(ss[0]), int(ss[2]), int(ss[4]), int(ss[6]))
                            min_y = min(int(ss[1]), int(ss[3]), int(ss[5]), int(ss[7]))
                            max_x = max(int(ss[0]), int(ss[2]), int(ss[4]), int(ss[6]))
                            max_y = max(int(ss[1]), int(ss[3]), int(ss[5]), int(ss[7]))
                            # print(min_x, min_y, max_x, max_y)
                            if (max_x - min_x) >= 1 or (max_y - min_y) >= 1:
                                count += 1
                                region = imgSrc.crop((min_x, min_y, max_x, max_y))
                                region.save(os.path.join(to_image_root, gt.split('.')[0]) + '_' + str(count) + '.jpg')
                                newline = gt.split('.')[0] + '_' + str(count) + '.jpg' + " " + ss[8] + '\n'
                                txt_write.writelines(newline)
            except:
                pass
#
def rotate_90():
    root = "/workspace/xwh/aster/train/CASIA-10k/all/"
    lists = os.listdir(os.path.join(root, 'images'))
    count = 0
    for image in lists:
        count += 1
        print(count)
        img = Image.open(os.path.join(root + 'images/', image))
        kuan = img.size[0]
        gao = img.size[1]
        # print(kuan, gao)
        if gao > (1.5*kuan):
            img = img.transpose(Image.ROTATE_90)
            img.save(os.path.join(root, 'image90/')+image)
        else:
            img.save(os.path.join(root, 'image90/')+image)

def file_numbers():
    root = "/workspace/xwh/aster/train/all_images/train.txt"
    count = 0
    with open(root, 'r') as rs:
        lines = rs.readlines()
        for line in lines:
            if line[:11] == "train_ReCTS":
                count += 1
                print(count)




if __name__ == '__main__':
    # find_blank()
    # find_label_length()
    # open_picture()
    # delete_lines()
    # txt_paixu()
    # tiqu_chinese()
    # copy_image()
    # rotate_90()
    # copy_image()
    file_numbers()