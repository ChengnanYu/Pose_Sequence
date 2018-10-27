import os.path
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_posenet_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import numpy

class UnalignedPoseNetDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        #nvm_file = os.path.join(self.root , 'reconstruction.nvm')
        #self.points, self.cam_points = parse_nvm_3d_points(nvm_file)

        split_file = os.path.join(self.root , 'dataset_'+opt.phase+'.txt')
        self.A_paths = numpy.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=3, usecols=(0))
        self.A_paths_jpg = [path.split('.')[0]+'.jpg' for path in self.A_paths]
        self.A_paths = [os.path.join(self.root, path) for path in self.A_paths]
        self.A_paths_jpg = numpy.asarray(self.A_paths_jpg)
        self.A_paths = numpy.asarray(self.A_paths)
        self.A_poses = numpy.loadtxt(split_file, dtype=float, delimiter=' ', skiprows=3, usecols=(1,2,3,4,5,6,7))
        self.mean_image = numpy.load(os.path.join(self.root , 'mean_image.npy'))

        self.A_size = len(self.A_paths)
        self.transform = get_posenet_transform(opt, self.mean_image)
        
        '''
        generate index matrix like [[0,1,2,...,sequence_length-1],
                                    [1,2,3,...,sequence_length],
                                                  ...
                                    [             ...          ]] if overlap_step==1
        each row means a sequence
        the whole matrix represent all possible sequence given sequence length and overlap step
        '''        
        axis1 = numpy.arange(0, opt.sequence_length)
        axis0 = numpy.arange(0, self.A_size,opt.overlap_step)
        sequence_matrix = numpy.meshgrid(axis1,axis0)[0]+numpy.meshgrid(axis1,axis0)[1]
        '''
        since the ground truth file contains several video sequences,
        there are some invalid sequence in the matrix
        e.g. [260...260+sequence_length-1], the 260th row in the ground truth file is seq1/frame00261.png, while next row is seq4/frame00001.png
        we have to delete this kind of invalid sequence
        to make things simple, the index of last frame of each video sequence are stored in the 3rd row
        these indecis are only vaild if they are in the last column of matrix
        '''
        spliter = numpy.genfromtxt(split_file, dtype=numpy.int, delimiter=' ', skip_header=2, skip_footer=self.A_size)
        for i in range(spliter.size):
            print(self.A_paths[spliter[i]%self.A_size]+'  '+self.A_paths[(spliter[i]+1)%self.A_size])
            del_row = numpy.where(sequence_matrix[:,:-1]==spliter[i])[0]
            sequence_matrix = numpy.delete(sequence_matrix,del_row,0)
        self.sequence_matrix = sequence_matrix
        self.num_sequence = self.sequence_matrix.shape[0]

    def __getitem__(self, index):
        seq = self.sequence_matrix[index,:]
        A_path = self.A_paths[seq]
        A_path = A_path.tolist()

        #points_3d = []
        A_path_jpg = self.A_paths_jpg[seq]
        A_path_jpg = A_path_jpg.tolist()
        #for i in range(self.opt.sequence_length):
        #    path_jpg = A_path_jpg[i]
        #    points_3d.append(self.points[self.cam_points[path_jpg],:])
        
        #get an image sequence
        A_seq = []
        for i in range(self.opt.sequence_length):
            path = A_path[i]
            img = Image.open(path).convert('RGB')
            A = self.transform(img)
            A_seq.append(A)
        A_seq = torch.stack(A_seq)
        
        A_pose = self.A_poses[seq]

        return {'A': A_seq, 'B': A_pose,
                'A_paths': A_path, 'A_paths_jpg':A_path_jpg}

    def __len__(self):
        return self.num_sequence

    def name(self):
        return 'UnalignedPoseNetDataset'
