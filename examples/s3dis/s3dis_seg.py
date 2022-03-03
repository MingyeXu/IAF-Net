# S3DIS Example with IAFNet

# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../../')

import argparse
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
from PIL import Image
import time
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
import utils.metrics as metrics

import scipy.io as sio

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.astype(np.float32), pts_dest.astype(np.float32), K, omp=True)
    print(indices.shape)
    if K==1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)



# Part dataset only for training / validation
class PartDatasetTrainVal():

    def __init__ (self, filelist, folder,
                    training=False, 
                    block_size=2,
                    npoints = 4096,
                    iteration_number = None, nocolor=False, jitter=0.4):

        self.training = training
        self.filelist = filelist
        self.folder = folder
        self.bs = block_size
        self.nocolor = nocolor
        
        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False
        self.number_of_run = 10

        self.jitter = jitter # 0.8 for more
        self.transform = transforms.ColorJitter(
            brightness=jitter,
            contrast=jitter,
            saturation=jitter)

    def __getitem__(self, index):

        folder = self.folder
        if self.training:
            index = random.randint(0, len(self.filelist)-1)
            dataset = self.filelist[index]
        else:
            dataset = self.filelist[index//self.number_of_run]

        filename_data = os.path.join(folder, dataset, 'xyzrgb.npy')
        xyzrgb = np.load(filename_data).astype(np.float32)

        # load labels
        filename_labels = os.path.join(folder, dataset, 'label.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_labels))
        labels = np.load(filename_labels).astype(int).flatten()

        # pick a random point
        pt_id = random.randint(0, xyzrgb.shape[0]-1)
        pt = xyzrgb[pt_id, :3]

        mask_x = np.logical_and(xyzrgb[:,0]<pt[0]+self.bs/2, xyzrgb[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(xyzrgb[:,1]<pt[1]+self.bs/2, xyzrgb[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = xyzrgb[mask]
        lbs = labels[mask]

        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]

        if self.nocolor:
            features = np.ones((pts.shape[0], 1))
        else:
            features = pts[:,3:]
            if self.training and self.jitter > 0:
                features = features.astype(np.uint8)
                features = np.array(self.transform( Image.fromarray(np.expand_dims(features, 0)) ))
                features = np.squeeze(features, 0)
            
            features = features.astype(np.float32)
            features = features / 255 - 0.5

        pts = pts[:,:3]
        center_point = np.mean(pts,axis = 0)
        pts = pts - center_point
        if self.training:
            pts = rotate_point_cloud_z(pts)
    
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations

# Part dataset only for testing
class PartDatasetTest():


    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<=pt[0]+bs/2, self.xyzrgb[:,0]>=pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<=pt[1]+bs/2, self.xyzrgb[:,1]>=pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, filename, folder,
                    block_size=2,
                    npoints = 4096,
                    min_pick_per_point = 1, test_step=0.5, nocolor=False):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.min_pick_per_point = min_pick_per_point
        self.nocolor = nocolor
        # load data
        self.filename = filename
        filename_data = os.path.join(folder, self.filename, 'xyzrgb.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_data))
        self.xyzrgb = np.load(filename_data)
        filename_labels = os.path.join(folder, self.filename, 'label.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_labels))
        self.labels = np.load(filename_labels).astype(int).flatten()

        step = test_step
        mini = self.xyzrgb[:,:2].min(0)
        discretized = ((self.xyzrgb[:,:2]-mini).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step + mini + step/2

    def __getitem__(self, index):

        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        if self.nocolor:
            features = np.ones((pts.shape[0], 1))
        else:
            features = pts[:,3:6] / 255 - 0.5
        pts = pts[:,:3].copy()
        center_point = np.mean(pts,axis = 0)
        pts = pts - center_point
        # convert to torch
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        # return len(self.pts)
        return self.pts.shape[0]


def get_model(model_name, input_channels, output_channels, args):
    from networks.SemSeg_IAF import IAFNET
    return IAFNET(args,output_channels =output_channels )


def train(args, flist_train, flist_test):
    N_CLASSES = 13

    # create the network
    print("Creating network...")
    if args.nocolor:
        net = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    else:
        net = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)
    net.cuda()
    net = torch.nn.DataParallel(net)
    if args.pretrain:
        net.load_state_dict(torch.load(os.path.join(args.savedir, "pretrain.pth")))
    
    print("parameters", count_parameters(net))

    print("Creating dataloader and optimizer...")
    ds = PartDatasetTrainVal(flist_train, args.rootdir,
                             training=True, block_size=args.blocksize,
                             npoints=args.npoints,iteration_number=args.batchsize*args.iter, nocolor=args.nocolor, jitter=args.jitter)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True,
                                        num_workers=args.threads
                                        )

    ds_val = PartDatasetTrainVal(flist_test, args.rootdir,
                             training=False, block_size=args.blocksize,
                             npoints=args.npoints, nocolor=args.nocolor)
    test_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batchsize, shuffle=False,
                                        num_workers=args.threads
                                        )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("done")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # create the root folder
    print("Creating results folder")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "{}_area{}_{}_nocolor{}_drop{}_{}".format(args.model, args.area, args.npoints, args.nocolor, args.drop, time_string))
    os.makedirs(root_folder, exist_ok=True)
    print("done at", root_folder)
    
    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    maxIOU = 0.0
    # iterate over epochs
    for epoch in range(args.nepochs):

        #######
        # training
        scheduler.step()
        net.train()

        lr = optimizer.param_groups[0]['lr']
        print('LearningRate:',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = 0
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch))
        for pts, features, seg in t:

            features = features.cuda()
            pts = pts.cuda()
            seg = seg.cuda()
            
            optimizer.zero_grad()
            if args.nocolor:
                outputs = net(pts,pts)
            else:
                outputs,out_1,out_2,out_3,out_4,out_5, seg_2,seg_3,seg_4,seg_5 = net(features, pts, seg)
            loss_1 =  F.cross_entropy(out_1.contiguous().view(-1, N_CLASSES), seg.contiguous().view(-1))  
            loss_2 =  F.cross_entropy(out_2.contiguous().view(-1, N_CLASSES), seg_2.view(-1))
            loss_3 =  F.cross_entropy(out_3.contiguous().view(-1, N_CLASSES), seg_3.view(-1))
            loss_4 =  F.cross_entropy(out_4.contiguous().view(-1, N_CLASSES), seg_4.view(-1)) 
            loss_5 =  F.cross_entropy(out_5.contiguous().view(-1, N_CLASSES), seg_5.view(-1)) 

            loss =  F.cross_entropy(outputs.contiguous().view(-1, N_CLASSES), seg.contiguous().view(-1)) + 0.1*(loss_5+loss_4+loss_3+loss_2+loss_1)
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
            cm += cm_

            oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
            aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
            iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), LOSS=wblue(f"{train_loss/cm.sum():.4e}"))


        ######
        ## validation
        net.eval()
        cm_test = np.zeros((N_CLASSES, N_CLASSES))
        test_loss = 0
        
        t = tqdm(test_loader, ncols=80, desc="  Test epoch {}".format(epoch))
        with torch.no_grad():
            for pts, features, seg in t:
                
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()
                
                
                if args.nocolor:
                    outputs = net(pts,pts)
                else:
                    outputs = net(features, pts, seg) 
                outputs = outputs[0]      
                loss =  F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1))

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                cm_test += cm_

                oa_val = f"{metrics.stats_overall_accuracy(cm_test):.5f}"
                aa_val = f"{metrics.stats_accuracy_per_class(cm_test)[0]:.5f}"
                iou_val = f"{metrics.stats_iou_per_class(cm_test)[0]:.5f}"

                test_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wgreen(oa_val), AA=wgreen(aa_val), IOU=wgreen(iou_val), LOSS=wgreen(f"{test_loss/cm_test.sum():.4e}"))

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))
        if maxIOU<float(iou_val):
            # save the model
            torch.save(net.state_dict(), os.path.join(root_folder, "state_dict"+str(iou_val)+".pth"))
            maxIOU = float(iou_val)
        # write the logs
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_val} {aa_val} {iou_val}\n")
        logs.flush()

    logs.close()

def test(args, flist_test):
    

    N_CLASSES = 13


    # create the network
    print("Creating network...")
    if args.nocolor:
        net = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    else:    
        net = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)

    net.cuda()
    net = torch.nn.DataParallel(net)


    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    # net.cuda()
    net.eval()
    print("parameters", count_parameters(net))

    for filename in flist_test:
        print(filename)
        ds = PartDatasetTest(filename, args.rootdir,
                            block_size=args.blocksize,
                            min_pick_per_point= args.npick,
                            npoints= args.npoints,
                            test_step=args.test_step,
                            nocolor=args.nocolor
                            )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=False,
                                        num_workers=args.threads
                                        )

        xyzrgb = ds.xyzrgb[:,:3]
        scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
        labels = ds.labels
        total_time = 0
        iter_nb = 0
        with torch.no_grad():
            t = tqdm(loader, ncols=80)
            for pts, features, indices in t:
                
                t1 = time.time()
                features = features.cuda()
                pts = pts.cuda()
                if args.nocolor:
                    outputs = net(pts,pts)
                else:
                    outputs = net(features, pts, indices)
                t2 = time.time()

                outputs = outputs[0]
                outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                scores[indices.cpu().numpy().ravel()] += outputs_np

                iter_nb +=1
                total_time += (t2-t1)
                t.set_postfix(time=f"{total_time/(iter_nb*args.batchsize):05e}")

        mask = np.logical_not(scores.sum(1)==0)
        scores = scores[mask]
        pts_src = xyzrgb[mask]

        # create the scores for all points
        scores = nearest_correspondance(pts_src, xyzrgb, scores, K=15)

        # compute softmax
        scores = scores - scores.max(axis=1)[:,None]
        scores = np.exp(scores) / np.exp(scores).sum(1)[:,None]
        scores = np.nan_to_num(scores)

        os.makedirs(os.path.join(args.savedir, filename), exist_ok=True)

        # saving labels
        save_fname = os.path.join(args.savedir, filename, "pred.txt")
        s = np.argsort(scores, axis=1)
        scores = scores.argmax(1)
        np.savetxt(save_fname,scores,fmt='%d')

        if args.savepts:
            save_fname = os.path.join(args.savedir, filename, "pts.txt")
            xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
            np.savetxt(save_fname,xyzrgb,fmt=['%.4f','%.4f','%.4f','%d'])
            np.savetxt(save_fname,xyzrgb,fmt=['%.4f','%.4f','%.4f','%d'])
            save_fname_mat = os.path.join(args.savedir, filename, "vis.mat")
            sio.savemat(save_fname_mat, {
                        'points': xyzrgb, 
                        'pred': np.expand_dims(s,1),
                        'labels':labels
                        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--rootdir", default='../../data/s3dis_data',type=str)
    parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--npoints", default=8192, type=int)
    parser.add_argument("--area", default=5, type=int)
    parser.add_argument("--blocksize", default=2, type=int)
    parser.add_argument("--iter", default=1000, type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--npick", default=16, type=int)
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--test_step", default=0.2, type=float)
    parser.add_argument("--nepochs", default=50, type=int)
    parser.add_argument("--jitter", default=0.4, type=float)
    parser.add_argument("--model", default="GS_Seg", type=str)
    parser.add_argument("--drop", default=0, type=float)
    args = parser.parse_args()


    # create the filelits (train / val) according to area
    print("Create filelist...", end="")
    filelist_train = []
    filelist_test = []
    for area_idx in range(1 ,7):
        folder = os.path.join(args.rootdir, f"Area_{area_idx}")
        datasets = [os.path.join(f"Area_{area_idx}", dataset) for dataset in os.listdir(folder)]
        if area_idx == args.area:
            filelist_test = filelist_test + datasets
        else:
            filelist_train = filelist_train + datasets
    filelist_train.sort()
    filelist_test.sort()
    print(f"done, {len(filelist_train)} train files, {len(filelist_test)} test files")


    if args.test:
        test(args, filelist_test)
    else:
        train(args, filelist_train, filelist_test)


if __name__ == '__main__':
    main()
