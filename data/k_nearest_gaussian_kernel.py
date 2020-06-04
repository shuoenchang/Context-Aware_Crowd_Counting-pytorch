import numpy as np
import scipy
import scipy.spatial
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
from tqdm.auto import tqdm

def better_gaussian_filter_density(img,points):
    img_shape=[img.shape[0],img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 5:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(img_shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(img,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print ('generate density...')
    pt2d = np.zeros(img_shape, dtype=np.float32)
    sigma = 0
    for i, pt in enumerate(points):
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 5:
            sigma += (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(img_shape))/2./2. #case: 1 point
    sigma /= gt_count
    density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    # root = '/mnt/sdc/final/data/ShanghaiTech/part_A'
    # root = '/mnt/sdc/final/data/ShanghaiTech/part_B'
    # root = '/mnt/sdc/final/data/QNRF'
    root = '/mnt/sdc/final/data/NWPU'
    
    # now generate the ShanghaiA's ground truth
    train_img_path = os.path.join(root,'train_data','images')
    val_img_path = os.path.join(root,'val_data','images')
    # part_B_train = os.path.join(root,'part_B_final/train_data','images')
    # part_B_test = os.path.join(root,'part_B_final/val_data','images')
    path_sets = [train_img_path, val_img_path]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
            
    for img_path in (img_paths):
        if os.path.exists(img_path.replace('.jpg','.npz').replace('images','density')):
            continue
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        img= plt.imread(img_path)#768行*1024列
        k = np.zeros((img.shape[0],img.shape[1]))
        # points = mat["image_info"][0,0][0,0][0] #1546person*2(col,row)
        points = mat["annPoints"] #1546person*2(col,row)
        k = gaussian_filter_density(img,points)
        # plt.imshow(k,cmap=CM.jet)
        # save density_map to disk
        print(np.sum(k))
        if(len(points)>5 and int(np.sum(k))/len(points) < 0.75):
            print("better_gaussian")
            k = better_gaussian_filter_density(img,points)
            print(np.sum(k))
        np.savez_compressed(img_path.replace('.jpg','.npz').replace('images','density'), k)
    
    # '''
    #now see a sample from ShanghaiA
    plt.imshow(Image.open(img_paths[0]))
    
    gt_file = np.load(img_paths[0].replace('.jpg','.npz').replace('images','density'))['arr_0']
    plt.imshow(gt_file,cmap=CM.jet)
    plt.show()
    print(np.sum(gt_file))# don't mind this slight variation
    
    # '''