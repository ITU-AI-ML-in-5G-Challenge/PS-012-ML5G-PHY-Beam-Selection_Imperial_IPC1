import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from ignite.metrics import TopKCategoricalAccuracy


from models import Lidar2D
from dataloader import LidarDataset2D

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--lidar_test_data", nargs='+', type=str, help="LIDAR test data file, if you want to merge multiple"
                                                                   " datasets, simply provide a list of paths, as follows:"
                                                                   " --lidar_training_data path_a.npz path_b.npz")
parser.add_argument("--beam_test_data", nargs='+', type=str, default=None,
                    help="Beam test data file, if you want to merge multiple"
                         " datasets, simply provide a list of paths, as follows:"
                         " --beam_training_data path_a.npz path_b.npz")
parser.add_argument("--model_path", type=str, help="Path, where the model is saved")
parser.add_argument("--preds_csv_path", type=str, default="unnamed_preds.csv",
                    help="Path, where the .csv file with the predictions will be saved")

args = parser.parse_args()


def evaluate(net, test_dataloader):

    with torch.no_grad():
        net.eval()
        preds_all = torch.empty((len(test_dataloader), 256))
        top_1 = TopKCategoricalAccuracy(k=1)
        top_5 = TopKCategoricalAccuracy(k=5)
        top_10 = TopKCategoricalAccuracy(k=10)
        for i, data in enumerate(test_dataloader):
            lidar, beams = data
            lidar = lidar.cuda()
            beams = beams.cuda()
            preds = net(lidar)
            preds = F.softmax(preds, dim=1)
            preds_all[i, :] = preds
            top_1.update((preds, torch.argmax(beams)))
            top_5.update((preds, torch.argmax(beams)))
            top_10.update((preds, torch.argmax(beams)))
        net.train()

        print("Top-1: {:.4f} Top-5: {:.4f} Top-10: {:.4f}".format(top_1.compute(), top_5.compute(), top_10.compute()))
    return preds_all


if __name__ == '__main__':

    test_dataset = LidarDataset2D(args.lidar_test_data, args.beam_test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Lidar2D()
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()

    preds_all = evaluate(model, test_dataloader).cpu().numpy()

    np.savetxt(args.preds_csv_path, preds_all, fmt='%.5f', delimiter=', ')
