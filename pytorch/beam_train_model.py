import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ignite.metrics import TopKCategoricalAccuracy

from models import Lidar2D
from dataloader import LidarDataset2D

import argparse


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


parser = argparse.ArgumentParser()

parser.add_argument("--lidar_training_data", nargs='+', type=str, help="LIDAR training data file, if you want to merge multiple"
                                                                       " datasets, simply provide a list of paths, as follows:"
                                                                       " --lidar_training_data path_a.npz path_b.npz")
parser.add_argument("--beam_training_data", nargs='+', type=str, help="Beam training data file, if you want to merge multiple"
                                                                      " datasets, simply provide a list of paths, as follows:"
                                                                      " --beam_training_data path_a.npz path_b.npz")
parser.add_argument("--lidar_validation_data", nargs='+', type=str, help="LIDAR validation data file, if you want to merge multiple"
                                                                         " datasets, simply provide a list of paths, as follows:"
                                                                         " --lidar_test_data path_a.npz path_b.npz")
parser.add_argument("--beam_validation_data", nargs='+', type=str, help="Beam validation data file, if you want to merge multiple"
                                                                        " datasets, simply provide a list of paths, as follows:"
                                                                        " --beam_test_data path_a.npz path_b.npz")
parser.add_argument("--model_path", type=str, default='test_model', help="Path, where the trained model will be saved")

args = parser.parse_args()


if __name__ == '__main__':

    train_dataset = LidarDataset2D(args.lidar_training_data, args.beam_training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    if args.lidar_validation_data is None and args.beam_validation_data is None:
        args.lidar_validation_data = args.lidar_training_data
        args.beam_validation_data = args.beam_training_data

    validation_dataset = LidarDataset2D(args.lidar_validation_data, args.beam_validation_data)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = Lidar2D().cuda()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    criterion = lambda y_pred, y_true: -torch.sum(torch.mean(y_true[y_pred>0] * torch.log(y_pred[y_pred>0]), axis=0))

    best_acc = 0.0
    for i in range(20):
        accumulated_loss = []
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            lidar, beams = data
            lidar = lidar.cuda()
            beams = beams.cuda()
            preds = model(lidar)
            preds = F.softmax(preds, dim=1)
            loss = criterion(preds, beams)
            loss.backward()
            optimizer.step()
            accumulated_loss.append(loss.item())
        scheduler.step()
        evaluate(model, validation_dataloader)

    torch.save(model.state_dict(), args.model_path)

