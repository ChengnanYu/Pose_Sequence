import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports one batch
opt.overlap_step = opt.sequence_length #no overlap during testing
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.rname = opt.name + '_all'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#test sequences = %d' % dataset_size)

# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
results_dir = os.path.join(opt.results_dir, opt.rname)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

testepochs = ['latest']
besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ
testepochs = numpy.arange(5, 1001, 5)
testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch medX  medQ\n')
testfile.write('==================\n')

model = create_model(opt)
visualizer = Visualizer(opt)

for testepoch in testepochs:
    opt.which_epoch = testepoch
    model.load_network(model.netG, 'G', opt.which_epoch)
    visualizer.change_log_path(opt.which_epoch)
    # test
    # err_pos = []
    # err_ori = []
    err = []
    print("epoch: "+ str(opt.which_epoch))
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()
        pose = model.get_current_pose()
        for j in range(img_path.shape[1]):
            print('\t%04d/%04d: process image sequence ... %s' % (i+1, len(dataset), img_path[0,j]), end='\r')
            image_path = img_path[0,j].split('/')[-2] + '/' + img_path[0,j].split('/')[-1]
            visualizer.save_estimated_pose(image_path, pose[0,j,:])
            # err_pos.append(err_p)
            # err_ori.append(err_o)
        err = err + model.get_current_errors()

    median_pos = numpy.median(err, axis=0)
    if median_pos[0] < besterror[1]:
        besterror = [opt.which_epoch, median_pos[0], median_pos[1]]
    print()
    # print("median position: {0:.2f}".format(numpy.median(err_pos)))
    # print("median orientat: {0:.2f}".format(numpy.median(err_ori)))
    print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
    testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(testepoch,
                                                     median_pos[0],
                                                     median_pos[1]))
    testfile.flush()
print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('==================\n')
testfile.close()
