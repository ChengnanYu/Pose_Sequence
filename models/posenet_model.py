import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from data.nvm import parse_nvm_3d_points
import pickle

torch.set_printoptions(threshold=10000)

class PoseNetModel(BaseModel):
    def name(self):
        return 'PoseNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # nvm file and 3d points
        nvm_file = os.path.join(opt.dataroot , 'reconstruction.nvm')
        self.points, self.cam_points = parse_nvm_3d_points(nvm_file)
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.sequence_length, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.sequence_length, opt.output_nc)

        # load/define networks
        googlenet_weights = None
        if self.isTrain and opt.init_weights != '':
            googlenet_file = open(opt.init_weights, "rb")
            googlenet_weights = pickle.load(googlenet_file, encoding="bytes")
            googlenet_file.close()
            print('initializing the weights from '+ opt.init_weights)
        self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))
        self.sequence_length = opt.sequence_length
        self.netG = networks.define_G(opt.input_nc, None, None, opt.which_model_netG, opt.sx, opt.sq, sequence_length=opt.sequence_length,
                                      init_from=googlenet_weights, isTest=not self.isTrain,
                                      gpu_ids = self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterion = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=opt.weight_decay,
                                                betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.image_paths = input['A_paths']
        self.image_paths_jpg = input['A_paths_jpg']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A = self.input_A.view(-1,self.input_A.size(2),self.input_A.size(3),self.input_A.size(4))
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        self.pred_B = self.netG(self.input_A)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return np.transpose(np.asarray(self.image_paths))

    def qmul(self, q, r):
        """
        code borrow from QuaterNet, auther: facebookresearch
        github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4
        
        original_shape = q.shape
        
        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
        
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    def qinv(self, q):
        """
        Inverts quaternions
        :param q: N x sequence length x 4
        :return: q*: N x sequence length x 4 
        """
        q_inv = torch.cat((q[:, :, :1], -q[:, :, 1:]), dim=2)
        return q_inv

    def quat2mat(self, quat):
        """
        Convert unit quaternion to rotation matrix.
        Args:
        quat: size = [batch size, sequence length, 4]
        Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch size, sequence length, 3, 3]
        """
        w, x, y, z = quat[:,:,0], quat[:,:,1], quat[:,:,2], quat[:,:,3]

        batchsize = quat.size(0)
        sequence_length = quat.size(1)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                              2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                              2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=2).reshape(batchsize, sequence_length, 3, 3)
        return rotMat

    def backward_G(self):
        self.loss_G = 0
        self.loss_aux = np.array([0, 0, 0], dtype=np.float)
        image_paths_jpg = np.transpose(np.asarray(self.image_paths_jpg))

        mse_pos = self.criterion(self.pred_B[0], self.input_B[:, :, 0:3])
        ori_gt = F.normalize(self.input_B[:, :, 3:], p=2, dim=2)
        mse_ori = self.criterion(self.pred_B[1], ori_gt)

        ori_gt_rotMat = self.quat2mat(ori_gt)
        pred_B_rotMat = self.quat2mat(self.pred_B[1])
        for i in range(ori_gt.size(0)): # batch size
            for j in range(ori_gt.size(1)): #sequence length
                #print(image_paths_jpg[i,j])
                # for each image, get visible 3d points
                visible_points = torch.from_numpy(np.transpose(self.points[self.cam_points[image_paths_jpg[i,j]],:])).float()
                visible_points = visible_points.to(ori_gt.device)
                # reprojected coordinates = R*g + x, equation (6) in the paper
                reproj_gt = torch.mm(ori_gt_rotMat[i,j,:,:],visible_points)+torch.unsqueeze(self.input_B[i,j,0:3],1)
                # (u,v) = (u'/w', v'/w'), equation (6) in the paper
                reproj_gt = reproj_gt/reproj_gt[2,:]
                reproj_gt = reproj_gt[:2,:]
                #print(reproj_gt)
                reproj_pred = torch.mm(pred_B_rotMat[i,j,:,:],visible_points)+torch.unsqueeze(self.pred_B[0][i,j,:],1)
                reproj_pred = reproj_pred/reproj_pred[2,:]
                reproj_pred = reproj_pred[:2,:]
                #print(reproj_pred)
                mse_reporj = self.criterion(reproj_pred,reproj_gt)
                #print(mse_reporj.item())
                self.loss_G = self.loss_G + mse_reporj
            #print()
        self.loss_G = self.loss_G/(ori_gt.size(0)*ori_gt.size(1))
        #self.loss_G = torch.exp(-self.netG.sx) * mse_pos + self.netG.sx + torch.exp(-self.netG.sq) * (mse_ori) + self.netG.sq
        self.loss_aux[2] = self.loss_G.item()
        self.loss_aux[0] = mse_pos.item()
        self.loss_aux[1] = mse_ori.item()
        #mse_pos = self.criterion(self.pred_B[0], self.input_B[:, :, 0:3])
        #ori_gt = F.normalize(self.input_B[:, :, 3:], p=2, dim=2)
        #mse_ori = self.criterion(self.pred_B[1], ori_gt) * self.opt.beta
        #self.loss_G = mse_pos + mse_ori
        #self.loss_aux[0] = mse_pos.item()
        #self.loss_aux[1] = mse_ori.item()
        #self.loss_aux[2] = mse_pos.item() + mse_ori.item()
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            # return OrderedDict([('G_aux1', self.loss_aux[0]),
            #                     ('G_aux2', self.loss_aux[1] - self.loss_aux[0]),
            #                     ('G_final', self.loss_aux[2] - self.loss_aux[1].sum()),
            #                     ('mse_pos_final', self.loss_aux[3]),
            #                     ('mse_ori_final', self.loss_aux[4]),
            #                     ])
            return OrderedDict([('reproj_loss', self.loss_aux[2]),
                                ('mse_pos', self.loss_aux[0]),
                                ('mse_ori', self.loss_aux[1]),])
        pred_xyz = self.pred_B[0].detach().cpu().numpy() #shape (batch size, sequence length, 3)
        pred_xyz = pred_xyz.reshape(pred_xyz.shape[0]*pred_xyz.shape[1],pred_xyz.shape[2])
        target_xyz = self.input_B[:, :, 0:3].detach().cpu().numpy()
        target_xyz = target_xyz.reshape(target_xyz.shape[0]*target_xyz.shape[1],target_xyz.shape[2])
        error_xyz = np.linalg.norm(pred_xyz-target_xyz, axis=1, keepdims=True)
        
        ori_gt = F.normalize(self.input_B[:, :, 3:], p=2, dim=2)
        ori_target = ori_gt.detach().cpu().numpy()
        ori_target = ori_target.reshape(ori_target.shape[0]*ori_target.shape[1],ori_target.shape[2])
        ori_pred = self.pred_B[1].detach().cpu().numpy()
        ori_pred = ori_pred.reshape(ori_pred.shape[0]*ori_pred.shape[1],ori_pred.shape[2])
        abs_distance = np.abs(np.sum(np.multiply(ori_target,ori_pred),axis=1,keepdims=True))
        ori_err = 2*(180/np.pi)*np.arccos(abs_distance)
        
        err = np.concatenate((error_xyz,ori_err),axis=1)
        err = err.tolist()
        #pos_err = torch.dist(self.pred_B[0], self.input_B[:, :, 0:3])
        #ori_gt = F.normalize(self.input_B[:, :, 3:], p=2, dim=2)
        #abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        # abs_distance = torch.clamp(abs_distance, max=1)
        #ori_err = 2*180/np.pi* torch.acos(abs_distance)
        return err

    def get_middle_errors(self):
        if self.opt.isTrain:
            # return OrderedDict([('G_aux1', self.loss_aux[0]),
            #                     ('G_aux2', self.loss_aux[1] - self.loss_aux[0]),
            #                     ('G_final', self.loss_aux[2] - self.loss_aux[1].sum()),
            #                     ('mse_pos_final', self.loss_aux[3]),
            #                     ('mse_ori_final', self.loss_aux[4]),
            #                     ])
            return OrderedDict([('total_loss', self.loss_aux[2]),
                                ('mse_pos_final', self.loss_aux[0]),
                                ('mse_ori_final', self.loss_aux[1])])
        pred_xyz = self.pred_B[0][:, self.sequence_length//2, :].detach().cpu().numpy() #shape (batch size, 3)
        #pred_xyz = pred_xyz.reshape(pred_xyz.shape[0]*pred_xyz.shape[1],pred_xyz.shape[2])
        target_xyz = self.input_B[:, self.sequence_length//2, 0:3].detach().cpu().numpy()
        #target_xyz = target_xyz.reshape(target_xyz.shape[0]*target_xyz.shape[1],target_xyz.shape[2])
        error_xyz = np.linalg.norm(pred_xyz-target_xyz, axis=1, keepdims=True)
        
        ori_gt = F.normalize(self.input_B[:, self.sequence_length//2, 3:], p=2, dim=1)
        ori_target = ori_gt.detach().cpu().numpy()
        #ori_target = ori_target.reshape(ori_target.shape[0]*ori_target.shape[1],ori_target.shape[2])
        ori_pred = self.pred_B[1][:, self.sequence_length//2, :].detach().cpu().numpy()
        #ori_pred = ori_pred.reshape(ori_pred.shape[0]*ori_pred.shape[1],ori_pred.shape[2])
        abs_distance = np.abs(np.sum(np.multiply(ori_target,ori_pred),axis=1,keepdims=True))
        ori_err = 2*(180/np.pi)*np.arccos(abs_distance)
        
        err = np.concatenate((error_xyz,ori_err),axis=1)
        err = err.tolist()
        #pos_err = torch.dist(self.pred_B[0], self.input_B[:, :, 0:3])
        #ori_gt = F.normalize(self.input_B[:, :, 3:], p=2, dim=2)
        #abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        # abs_distance = torch.clamp(abs_distance, max=1)
        #ori_err = 2*180/np.pi* torch.acos(abs_distance)
        return err

    def get_middle9_errors(self):
        if self.opt.isTrain:
            # return OrderedDict([('G_aux1', self.loss_aux[0]),
            #                     ('G_aux2', self.loss_aux[1] - self.loss_aux[0]),
            #                     ('G_final', self.loss_aux[2] - self.loss_aux[1].sum()),
            #                     ('mse_pos_final', self.loss_aux[3]),
            #                     ('mse_ori_final', self.loss_aux[4]),
            #                     ])
            return OrderedDict([('total_loss', self.loss_aux[2]),
                                ('mse_pos_final', self.loss_aux[0]),
                                ('mse_ori_final', self.loss_aux[1]),
                                ])
        pred_xyz = self.pred_B[0][:, (self.sequence_length//2-4):(self.sequence_length//2+5), :].detach().cpu().numpy() #shape (batch size,3,3)
        pred_xyz = pred_xyz.reshape(pred_xyz.shape[0]*pred_xyz.shape[1],pred_xyz.shape[2])
        target_xyz = self.input_B[:, (self.sequence_length//2-4):(self.sequence_length//2+5), 0:3].detach().cpu().numpy()
        target_xyz = target_xyz.reshape(target_xyz.shape[0]*target_xyz.shape[1],target_xyz.shape[2])
        error_xyz = np.linalg.norm(pred_xyz-target_xyz, axis=1, keepdims=True)
        
        ori_gt = F.normalize(self.input_B[:, (self.sequence_length//2-4):(self.sequence_length//2+5), 3:], p=2, dim=2)
        ori_target = ori_gt.detach().cpu().numpy()
        ori_target = ori_target.reshape(ori_target.shape[0]*ori_target.shape[1],ori_target.shape[2])
        ori_pred = self.pred_B[1][:, (self.sequence_length//2-4):(self.sequence_length//2+5), :].detach().cpu().numpy()
        ori_pred = ori_pred.reshape(ori_pred.shape[0]*ori_pred.shape[1],ori_pred.shape[2])
        abs_distance = np.abs(np.sum(np.multiply(ori_target,ori_pred),axis=1,keepdims=True))
        ori_err = 2*(180/np.pi)*np.arccos(abs_distance)
        
        err = np.concatenate((error_xyz,ori_err),axis=1)
        err = err.tolist()
        #pos_err = torch.dist(self.pred_B[0], self.input_B[:, :, 0:3])
        #ori_gt = F.normalize(self.input_B[:, :, 3:], p=2, dim=2)
        #abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        # abs_distance = torch.clamp(abs_distance, max=1)
        #ori_err = 2*180/np.pi* torch.acos(abs_distance)
        return err

    def get_middle15_errors(self):
        if self.opt.isTrain:
            # return OrderedDict([('G_aux1', self.loss_aux[0]),
            #                     ('G_aux2', self.loss_aux[1] - self.loss_aux[0]),
            #                     ('G_final', self.loss_aux[2] - self.loss_aux[1].sum()),
            #                     ('mse_pos_final', self.loss_aux[3]),
            #                     ('mse_ori_final', self.loss_aux[4]),
            #                     ])
            return OrderedDict([('total_loss', self.loss_aux[2]),
                                ('mse_pos_final', self.loss_aux[0]),
                                ('mse_ori_final', self.loss_aux[1]),
                                ])
        pred_xyz = self.pred_B[0][:, (self.sequence_length//2-7):(self.sequence_length//2+8), :].detach().cpu().numpy() #shape (batch size,3,3)
        pred_xyz = pred_xyz.reshape(pred_xyz.shape[0]*pred_xyz.shape[1],pred_xyz.shape[2])
        target_xyz = self.input_B[:, (self.sequence_length//2-7):(self.sequence_length//2+8), 0:3].detach().cpu().numpy()
        target_xyz = target_xyz.reshape(target_xyz.shape[0]*target_xyz.shape[1],target_xyz.shape[2])
        error_xyz = np.linalg.norm(pred_xyz-target_xyz, axis=1, keepdims=True)
        
        ori_gt = F.normalize(self.input_B[:, (self.sequence_length//2-7):(self.sequence_length//2+8), 3:], p=2, dim=2)
        ori_target = ori_gt.detach().cpu().numpy()
        ori_target = ori_target.reshape(ori_target.shape[0]*ori_target.shape[1],ori_target.shape[2])
        ori_pred = self.pred_B[1][:, (self.sequence_length//2-7):(self.sequence_length//2+8), :].detach().cpu().numpy()
        ori_pred = ori_pred.reshape(ori_pred.shape[0]*ori_pred.shape[1],ori_pred.shape[2])
        abs_distance = np.abs(np.sum(np.multiply(ori_target,ori_pred),axis=1,keepdims=True))
        ori_err = 2*(180/np.pi)*np.arccos(abs_distance)
        
        err = np.concatenate((error_xyz,ori_err),axis=1)
        err = err.tolist()
        #pos_err = torch.dist(self.pred_B[0], self.input_B[:, :, 0:3])
        #ori_gt = F.normalize(self.input_B[:, :, 3:], p=2, dim=2)
        #abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        # abs_distance = torch.clamp(abs_distance, max=1)
        #ori_err = 2*180/np.pi* torch.acos(abs_distance)
        return err

    def get_current_pose(self):
        pose = np.concatenate((self.pred_B[0].detach().cpu().numpy(),self.pred_B[1].detach().cpu().numpy()),axis=2)
        #pose = pose.reshape(pose.shape[0]*pose.shape[1],pose.shape[2])
        return pose

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # input_B = util.tensor2im(self.input_B.data)
        return OrderedDict([('input_A', input_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
