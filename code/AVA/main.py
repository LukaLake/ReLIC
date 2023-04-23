import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torchvision.datasets.folder import default_loader

# from models.u_model import NIMA
# from models.e_model import NIMA
# from models.relic_model import NIMA
from models.relic1_model import NIMA
# from models.relic2_model import NIMA
from dataset import AVADataset
current_path = os.path.abspath(__file__)
directory_name = os.path.dirname(current_path)
import sys
sys.path.append("..")
from util import EDMLoss,AverageMeter
import option as option
import shutil

filepath='../../data/AVA/checkpoint/log_test.txt'
f = open(os.path.join(directory_name, filepath), 'w')

opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))


def adjust_learning_rate(params, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = params.init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_score(opt,y_pred):
    w = torch.from_numpy(np.linspace(1,10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, 'train.csv')
    val_csv_path = os.path.join(opt.path_to_save_csv, 'val.csv')
    test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')

    train_ds = AVADataset(train_csv_path, opt.path_to_images, if_train = True)
    val_ds = AVADataset(val_csv_path, opt.path_to_images, if_train = False)
    test_ds = AVADataset(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader,test_loader


def train(opt,model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        y = y.to(opt.device)
        y_pred = model(x)
        loss = criterion(p_target=y, p_estimate=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/train_loss.avg", train_losses.avg, global_step=global_step + idx)
    return train_losses.avg


def validate(opt,model, loader, criterion, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        y = y.type(torch.FloatTensor)
        y = y.to(opt.device)

        y_pred = model(x)
        pscore, pscore_np = get_score(opt,y_pred)
        tscore, tscore_np = get_score(opt,y)

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()

        loss = criterion(p_target=y, p_estimate=y_pred)
        validate_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)

    lcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    print('lcc_mean', lcc_mean[0])
    print('srcc_mean', srcc_mean[0])

    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)
    print(acc)

    return validate_losses.avg, acc, lcc_mean, srcc_mean


def start_train(opt):
    train_loader, val_loader, test_loader = create_data_part(opt)
    model = NIMA()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.init_lr)
    criterion = EDMLoss()
    model = model.to(opt.device)
    criterion.to(opt.device)

    writer = SummaryWriter(log_dir=os.path.join(opt.experiment_dir_name, 'logs'))

    for e in range(opt.num_epoch):
        adjust_learning_rate(opt, optimizer, e)
        train_loss = train(opt,model=model, loader=train_loader, optimizer=optimizer, criterion=criterion,
                           writer=writer, global_step=len(train_loader) * e,
                           name=f"{opt.experiment_dir_name}_by_batch")
        val_loss,vacc,vlcc,vsrcc = validate(opt,model=model, loader=val_loader, criterion=criterion,
                            writer=writer, global_step=len(val_loader) * e,
                            name=f"{opt.experiment_dir_name}_by_batch")
        test_loss, tacc, tlcc, tsrcc = validate(opt, model=model, loader=test_loader, criterion=criterion,
                                               writer=writer, global_step=len(val_loader) * e,
                                               name=f"{opt.experiment_dir_name}_by_batch")
        model_name = f"epoch_{e}_.pth"
        torch.save(model.state_dict(), os.path.join(opt.experiment_dir_name, model_name))

        f.write(
            'epoch:%d,v_lcc:%.4f,v_srcc:%.4f,v_acc:%.5f,tlcc:%.4f,t_srcc:%.4f,t_acc:%.5f,train:%.5f,val:%.5f,test:%.5f\r\n'
            % (e, vlcc[0], vsrcc[0], vacc, tlcc[0], tsrcc[0], tacc, train_loss, val_loss, test_loss))
        f.flush()

        writer.add_scalars("epoch_loss", {'train': train_loss, 'val': val_loss, 'test': test_loss},
                           global_step=e)

        writer.add_scalars("lcc_srcc", {'val_lcc': vlcc[0], 'val_srcc': vsrcc[0],
                                        'test_lcc': tlcc[0], 'test_srcc': tsrcc[0]},
                           global_step=e)

        writer.add_scalars("acc",{'val_acc': vacc, 'test_acc': tacc}, global_step=e)

    writer.close()
    f.close()


def start_check_model(opt):
    _,val_loader, test_loader = create_data_part(opt)
    model = NIMA()
    model.eval()
    model.load_state_dict(torch.load(opt.path_to_model_weight,map_location='cuda:0'))
    criterion = EDMLoss()

    model = model.to(opt.device)
    criterion.to(opt.device)

    test_loss, acc, lcc_mean, srcc_mean = validate(opt,model=model, loader=test_loader, criterion=criterion)
    val_loss, vacc, vlcc_mean, vsrcc_mean = validate(opt,model=model, loader=val_loader, criterion=criterion)

    print('loss:', test_loss, 'acc:', acc, 'lcc:', lcc_mean[0], 'srcc:', srcc_mean[0])
    print('vloss:', val_loss, 'vacc:', vacc, 'vlcc:', vlcc_mean[0], 'vsrcc:', vsrcc_mean[0])

# 预测单个图片
def pred_single(opt):
    model = NIMA()
    model.eval()
    model.load_state_dict(torch.load(opt.path_to_model_weight,map_location='cuda:0'))
    criterion = EDMLoss()

    model = model.to(opt.device)
    criterion.to(opt.device)

    # test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')
    # ds = AVADataset(test_csv_path, opt.path_to_images, if_train=False)
    # loader = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
    image_path=os.path.join("C:/Users/123/Documents/GitHub/ReLIC/data/AVA/Batch1", opt.image_name)
    
    image = default_loader(image_path)  # 读取为Image对象
    x=transform(image)
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x).float()
    x = x.to(opt.device)
    y_pred = model(x)

    pscore, pscore_np = get_score(opt,y_pred)
    print('y_pred:', y_pred)
    print(pscore_np)

# 预测多个图片
def pred(opt):

    path = opt.path_to_images+'/group1'
    image_filenames = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_filenames.append(filename)


    model = NIMA()
    model.eval()
    model.load_state_dict(torch.load(opt.path_to_model_weight,map_location='cuda:0'))
    criterion = EDMLoss()

    model = model.to(opt.device)
    criterion.to(opt.device)

    # test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')
    # ds = AVADataset(test_csv_path, opt.path_to_images, if_train=False)
    # loader = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
    image_score=[]

    for (image) in enumerate(tqdm(image_filenames)) :
        image_name=image[1]
        image_path=opt.path_to_images+'/group1/'+image_name
        
        image = default_loader(image_path)  # 读取为Image对象
        x=transform(image)
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).float()
        x = x.to(opt.device)
        y_pred = model(x)

        pscore, pscore_np = get_score(opt,y_pred)
        image_score.append(dict(image_name=image_name,score=pscore_np))

    image_good=[]
    for image in image_score:
        if image.get('score')>5.18:
            image_good.append(image)

    print(len(image_score))
    print(len(image_good))

    path_to_good=opt.path_to_images+'/good/'
    for image in enumerate(tqdm(image_good)):
        image_name=image[1].get('image_name')
        image_path=opt.path_to_images+'/group1/'+image_name
        shutil.copy(image_path,path_to_good+image_name)
            
        
if __name__ =="__main__":

    #### train model
    # start_train(opt)
    #### test model
    # start_check_model(opt)

    #### pred
    pred_single(opt)
    # pred(opt)