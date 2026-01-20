# File Name: prediction.py
# E-mail: jiang_dj@zju.edu.cn
import rdkit
from graph_constructor import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
import torch
import pandas as pd
import os
from dgl.data.utils import split_dataset
from sklearn.metrics import mean_squared_error
import argparse

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    keys = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bg, bg3, Ys, Keys = batch
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            outputs = model(bg, bg3)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            keys.append(Keys)
    return true, pred, keys


lr = 10 ** -3.5
epochs = 5000
batch_size = 128
num_workers = 0
tolerance = 0.0
patience = 70
l2 = 10 ** -6
repetitions = 3
# paras for model
node_feat_size = 40
edge_feat_size_2d = 12
edge_feat_size_3d = 21
graph_feat_size = 128
num_layers = 2
outdim_g3 = 128
d_FC_layer, n_FC_layer = 128, 2
dropout = 0.2
n_tasks = 1
mark = '3d'
path_marker = '/'



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_ls_path', type=str, default='./examples/graph_ls_path',
                           help="absolute path for storing graph list objects")
    argparser.add_argument('--graph_dic_path', type=str, default='./examples/graph_dic_path',
                           help="absolute path for storing graph dictionary objects (temporary files)")
    argparser.add_argument('--model_path', type=str, default='./model/pdb2016_10A_20201230_3d_2_ciap1.pth',
                           help="absolute path for storing pretrained model")
    argparser.add_argument('--cpu', type=bool, default=True,
                           help="using cpu for the prediction (default:True)")
    argparser.add_argument('--gpuid', type=int, default=0,
                           help="the gpu id for the prediction")
    argparser.add_argument('--num_process', type=int, default=12,
                           help="the number of process for generating graph objects")
    argparser.add_argument('--input_path', type=str, default='./examples/ign_input',
                           help="the absoute path for storing ign input files")
    args = argparser.parse_args()
    graph_ls_path, graph_dic_path, model_path, cpu, gpuid, num_process, input_path = args.graph_ls_path, \
                                                                                     args.graph_dic_path, \
                                                                                     args.model_path, \
                                                                                     args.cpu, \
                                                                                     args.gpuid, \
                                                                                     args.num_process, \
                                                                                     args.input_path
    if not os.path.exists('./stats'):
        os.makedirs('./stats')

    # 불필요한 블록 삭제 후, 필요한 설정만 남김
    limit = None
# --- 여기부터 붙여넣기 시작 ---
    # dataset_name = os.path.basename(os.path.dirname(input_path))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(input_path)))
    print(dataset_name)
    # 1. answer.csv 파일을 DataFrame으로 읽어옵니다.
    sanitize_map = {
        "BDB2020+": "BDB2020+_sanitize_True",
        "EGFR": "EGFR_sanitize_True",
        "Mpro": "Mpro_sanitize_True",
        "LP-PDB": "LP_PDBBind_sanitize_True",
        "CASF2016": "CASF2016"
    }

    # AF3면 매핑, 아니면 그대로 사용
    if dataset_name in sanitize_map:
        answer_dataset = sanitize_map[dataset_name]
    else:
        answer_dataset = dataset_name

    answer_csv_path = os.path.join(
        "/NAS_Storage3/mjchung/IGN_preprocess/acutal_answer",
        answer_dataset,
        "stats",
        "answer.csv"
    )

    try:
        answer_df = pd.read_csv(answer_csv_path)
        print(f"Successfully loaded the label file: {answer_csv_path}")
        
        # 2. pdbid를 key로, pKa를 value로 하는 딕셔너리를 생성합니다. (훨씬 빠른 조회를 위함)
        # set_index('pdbid')로 pdbid를 인덱스로 만들고, ['pKa']로 pKa 컬럼을 선택한 뒤, to_dict()로 딕셔너리로 변환합니다.
        label_dict = answer_df.set_index('pdbid')['pKa'].to_dict()

    except FileNotFoundError:
        print(f"Error: {answer_csv_path} not found.")
        print("Labels will be set to 0.")
        label_dict = None


    keys = os.listdir(input_path)
    labels = []
    data_dirs = []
    
    for key in keys:
        data_dirs.append(input_path + path_marker + key)

        if label_dict is not None:
            # 3. 딕셔너리에서 key(pdbid)로 pKa 값을 바로 조회합니다.
            # .get() 메소드를 사용하면 key가 없을 경우 기본값(0)을 반환하여 오류를 방지합니다.
            label_value = label_dict.get(key)
            
            if label_value is not None:
                labels.append(label_value)
            else:
                print(f"Warning: Key '{key}' not found in {answer_csv_path}. Setting label to 0.")
                labels.append(0)
        else:
            # 파일을 로드하지 못했다면 0을 추가합니다.
            labels.append(0)

    # --- 여기까지 붙여넣기 끝 ---

    # generating the graph objective using multi process
    test_dataset = GraphDatasetV2MulPro(keys=keys[:limit], labels=labels[:limit], data_dirs=data_dirs[:limit],
                                        graph_ls_path=graph_ls_path,
                                        graph_dic_path=graph_dic_path,
                                        dis_threshold=12.0,
                                        num_process=num_process, path_marker=path_marker)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn_v2_MulPro)

    DTIModel = DTIPredictorV4_V2(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                 graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                 d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, n_tasks=n_tasks)
    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%s" % gpuid)
    DTIModel.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    DTIModel.to(device)

    test_true, test_pred, key = run_a_eval_epoch(DTIModel, test_dataloader, device)
    test_true = np.concatenate(np.array(test_true), 0).flatten()
    test_pred = np.concatenate(np.array(test_pred), 0).flatten()
    key = np.concatenate(np.array(key), 0).flatten()
    res = pd.DataFrame({'pdbid': key, 'pk': test_true, 'pred': test_pred})

    output_dir = "/NAS_Storage3/mjchung/af3_post_processing_minimum/data_nothing/CASF2016/test"
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"{dataset_name}.csv")
    res.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")