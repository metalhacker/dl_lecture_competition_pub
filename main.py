import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.models.transflownet import TransFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def deep_supervision_loss(flow_dict, gt_flow, weight_dict = {'flow3': 0.5, 'flow2': 0.25, 'flow1': 0.125, 'flow0': 0.0625}):
    loss3 = compute_epe_error(flow_dict['flow3'], gt_flow) * weight_dict['flow3']
    loss2 = compute_epe_error(F.interpolate(flow_dict['flow2'], size = flow_dict['flow3'].shape[2:], mode = 'bilinear'), gt_flow) * weight_dict['flow2']
    loss1 = compute_epe_error(F.interpolate(flow_dict['flow1'], size = flow_dict['flow3'].shape[2:], mode = 'bilinear'), gt_flow) * weight_dict['flow1']
    loss0 = compute_epe_error(F.interpolate(flow_dict['flow0'], size = flow_dict['flow3'].shape[2:], mode = 'bilinear'), gt_flow) * weight_dict['flow0']

    loss = loss3 + loss2 + loss1 + loss0

    return loss

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    # model = EVFlowNet(args.train).to(device)
    model = TransFlowNet(args.train).to(device)
    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    # ------------------
    #   Start training
    # ------------------
    model.train()
    for epoch in range(args.train.epochs):
        total_loss_epe = 0
        total_loss_deep = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            # flow = model(event_image) # [B, 2, 480, 640]
            flow, flow_dict = model(event_image) # [B, 2, 480, 640]

            loss_epe: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
            print(f"\nbatch {i} EPE loss: {loss_epe.item()}")


            loss_deep: torch.Tensor = deep_supervision_loss(flow_dict, ground_truth_flow)
            print(f"batch {i} Deep loss: {loss_deep.item()}")
            optimizer.zero_grad()
            loss_deep.backward()
            optimizer.step()


            total_loss_epe += loss_epe.item()
            total_loss_deep += loss_deep.item()

            # torch.cuda.empty_cache()

        print(f'Epoch {epoch+1}, EPE Loss: {total_loss_epe / len(train_data)}, Deep Loss: {total_loss_deep / len(train_data)}')

    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow, _ = model(event_image) # [1, 2, 480, 640]
            # flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
