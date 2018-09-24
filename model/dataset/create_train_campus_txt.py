import os

def gt_file(path, file_txt, replace_pattern=None):

    with open(file_txt, 'w') as f:
        for path, dirs, files in os.walk(path):
            for file in files:
                im_file = os.path.join(path,file)
                if "gt_" in im_file and 'road' in im_file:
                    line_gt = im_file.split('dataset/')[-1]

                    line_ori = line_gt
                    for pa in replace_pattern:                   
                        line_ori = line_ori.replace(pa,'')
                    line = line_ori + " " + line_gt + "\n"

                    f.write(line)

def gt_file_cityscapes(path, file_txt, replace_pattern=None):

    with open(file_txt, 'w') as f:
        for path, dirs, files in os.walk(path):
            for file in files:
                im_file = os.path.join(path,file)
                if "gtFine_color" in im_file and 'train' in im_file:
                    line_gt = im_file.split('dataset/')[-1]

                    line_ori = line_gt
                    line_ori = line_ori.replace('gtFine','leftImg8bit')
                    line_ori = line_ori.replace('_color','')
                    # for pa in replace_pattern:                   
                    #     line_ori = line_ori.replace(pa,'')
                    line = line_ori + " " + line_gt + "\n"

                    f.write(line)



def split_kitti(file_txt_kitti, val_txt, train_txt):

    with open(file_txt_kitti,'r') as f:

        for line in f:
        # f.readline()
            print line


if __name__ == "__main__":

    base_path = os.path.dirname(__file__)
    # campus_dir = os.path.join(base_path,'campus')

    # file_txt_campus = 'train_campus.txt'
    # file_txt_campus = os.path.join(base_path, file_txt_campus)
    # gt_file(campus_dir, file_txt_campus, ['gt_'])

    kitti_dir = os.path.join(base_path,'kitti')
    file_txt_kitti = 'train_kitti.txt'
    file_txt_kitti = os.path.join(base_path, file_txt_kitti)    
    gt_file(kitti_dir, file_txt_kitti, ['gt_','road_'])

    # split_kitti(file_txt_kitti)

    # file_txt_cityscapes = 'train_cityscapes.txt'
    # cityscapes_dir = os.path.join(base_path, 'cityscapes')
    # file_txt_cityscapes = os.path.join(base_path, file_txt_cityscapes)
    # gt_file_cityscapes(cityscapes_dir, file_txt_cityscapes)
