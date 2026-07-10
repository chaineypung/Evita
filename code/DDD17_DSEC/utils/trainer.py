from thop import profile
import torch
import os
import time
from model.evita import Evita
from utils.func import IOStream
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.scheduler import PolynomialLR, WarmupPolyLR, WarmupCosineLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.metrics import MetricsSemseg
from utils.labels import dataset_info
from torch import nn, Tensor
from torch.nn import functional as F

REDUCE_LR_SCHEDULERS = ['reduce01', 'reduce05']


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Trainer():

    def __init__(self, cfg):

        self.cfg = cfg

        self.writer = SummaryWriter(log_dir=self.cfg['TRAIN']['log_dir'])

        self.printer = IOStream(self.cfg['TRAIN']['log_dir'] + '/run.log')
        print(str(self.cfg))
        self.printer.cprint(str(self.cfg))

        semseg_ignore_label, semseg_class_names, _ = dataset_info(semseg_num_classes=self.cfg['DATASET']['classes'])

        if self.cfg['DATASET']['name'] == 'DDD17Event':
            from datasets.ddd17_dataset import DDD17Event
            self.training_dataset = DDD17Event(root=self.cfg['DATASET']['path'], split='train',
                                               event_representation=self.cfg['DATASET']['event_representation'],
                                               nr_events_data=1, delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                               nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                               require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                               augmentation=True, fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                               random_crop=True)
            self.validation_dataset = DDD17Event(root=self.cfg['DATASET']['path'], split='test',
                                                 event_representation=self.cfg['DATASET']['event_representation'],
                                                 nr_events_data=1, delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                                 nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                                 require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                                 augmentation=False, fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                                 random_crop=False)
            self.training_loader = DataLoader(self.training_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                              batch_size=self.cfg['TRAIN']['batch_size'], shuffle=True)
            self.validation_loader = DataLoader(self.validation_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                                batch_size=self.cfg['TRAIN']['batch_size'])

        elif self.cfg['DATASET']['name'] == 'DSECEvent':
            from datasets.dsec_dataset import DSECEvent
            self.training_dataset = DSECEvent(self.cfg['DATASET']['path'], nr_events_data=1,
                                              delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                              nr_events_window=self.cfg['DATASET']['nr_events'],
                                              augmentation=True,
                                              mode='train',
                                              event_representation=self.cfg['DATASET']['event_representation'],
                                              nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                              require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                              semseg_num_classes=self.cfg['DATASET']['classes'],
                                              fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                              random_crop=True)
            self.validation_dataset = DSECEvent(self.cfg['DATASET']['path'], nr_events_data=1,
                                                delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                                nr_events_window=self.cfg['DATASET']['nr_events'],
                                                augmentation=False,
                                                mode='val',
                                                event_representation=self.cfg['DATASET']['event_representation'],
                                                nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                                require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                                semseg_num_classes=self.cfg['DATASET']['classes'],
                                                fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                                random_crop=False)
            self.training_loader = DataLoader(self.training_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                              batch_size=self.cfg['TRAIN']['batch_size'], shuffle=True)
            self.validation_loader = DataLoader(self.validation_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                                batch_size=self.cfg['TRAIN']['batch_size'])

        self.model = Evita(num_classes=self.cfg['DATASET']['classes'],
                           backbone=self.cfg['MODEL']['name'],
                           pretrained_path=self.cfg['MODEL']['pretrained'])

        self.opt = AdamW(params=self.model.parameters(), lr=self.cfg['TRAIN']['lr_init'])
        self.cur_iter = 0
        max_epochs = int(self.cfg['TRAIN']['num_epochs'])
        total_iters = int(max_epochs * len(self.training_loader))
        warmup_iters = int(self.cfg['TRAIN']['warmup_iters'])
        assert warmup_iters <= total_iters, f"Warmup iterations should be less than total iterations, and the total iterations is {total_iters}."

        if self.cfg['TRAIN']['lr_scheduler'] == 'polynomial':
            self.scheduler = PolynomialLR(optimizer=self.opt, total_iters=total_iters)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'warmpoly':
            self.scheduler = WarmupPolyLR(optimizer=self.opt, T_max=total_iters, cur_iter=self.cur_iter,
                                          warmup_factor=1.0 / 3, warmup_iters=warmup_iters, power=0.8)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.opt, step_size=10, gamma=0.92,
                                                             last_epoch=-1, verbose='deprecated')
        elif self.cfg['TRAIN']['lr_scheduler'] == 'warmupcosine':
            self.scheduler = WarmupCosineLR(optimizer=self.opt, T_max=(max_epochs * total_iters))
        elif self.cfg['TRAIN']['lr_scheduler'] == 'reduce01':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='max', factor=0.1,
                                                                        patience=7, verbose=True)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'reduce05':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='max', factor=0.5,
                                                                        patience=5, verbose=True)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'fixed':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.opt, lr_lambda=lambda epoch: 1.0)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'pcx':
            import math
            for param_group in self.opt.param_groups:
                param_group['lr'] = 1.0
                param_group['initial_lr'] = 1.0

            steps_per_epoch = len(self.training_loader)
            warmup_steps = 5 * steps_per_epoch
            decay_steps = 75 * steps_per_epoch
            lr_start = 1e-5
            lr_peak = 9e-5
            lr_end = 1e-5

            def pcx_lambda(step):
                if step < warmup_steps:
                    alpha = step / warmup_steps
                    target_lr = lr_start + alpha * (lr_peak - lr_start)
                else:
                    steps_into_decay = step - warmup_steps
                    progress = min(steps_into_decay / decay_steps, 1.0)
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                    target_lr = lr_end + (lr_peak - lr_end) * cosine_factor
                return target_lr

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.opt, lr_lambda=pcx_lambda)

        self.criterion = OhemCrossEntropy(ignore_label=semseg_ignore_label)

        self.model.to(self.cfg['DEVICE'])

        self.metrics = MetricsSemseg(self.cfg['DATASET']['classes'], semseg_ignore_label, semseg_class_names)

        self.avg_top_k = int(self.cfg['TRAIN'].get('avg_top_k', 2))
        self.top_models = []

    def _update_top_models(self, epoch_id, mean_iou):
        state_dict_cpu = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }

        self.top_models.append({
            'epoch': epoch_id + 1,
            'mean_iou': mean_iou,
            'state_dict': state_dict_cpu
        })

        self.top_models = sorted(
            self.top_models,
            key=lambda x: x['mean_iou'],
            reverse=True
        )[:self.avg_top_k]

    def save_averaged_topk_model(self):
        if len(self.top_models) == 0:
            msg = "No model was collected for averaging."
            print(msg)
            self.printer.cprint(msg)
            return

        save_path = os.path.join(self.cfg['TRAIN']['log_dir'], 'best_modelv2.pth')

        if len(self.top_models) == 1:
            only_model = self.top_models[0]
            torch.save({
                'epoch': only_model['epoch'],
                'mean_iou': only_model['mean_iou'],
                'avg_top_k': 1,
                'state_dict': only_model['state_dict']
            }, save_path)

            msg = "Only one model collected. Save it as {}".format(save_path)
            print(msg)
            self.printer.cprint(msg)
            return

        avg_state_dict = {}

        for key in self.top_models[0]['state_dict'].keys():
            first_tensor = self.top_models[0]['state_dict'][key]

            if torch.is_floating_point(first_tensor):
                avg_tensor = first_tensor.clone()

                for model_info in self.top_models[1:]:
                    avg_tensor += model_info['state_dict'][key]

                avg_tensor /= float(len(self.top_models))
                avg_state_dict[key] = avg_tensor
            else:
                avg_state_dict[key] = first_tensor.clone()

        epochs = [model_info['epoch'] for model_info in self.top_models]
        mean_ious = [model_info['mean_iou'] for model_info in self.top_models]

        torch.save({
            'epoch': epochs,
            'mean_iou': mean_ious,
            'avg_top_k': len(self.top_models),
            'state_dict': avg_state_dict
        }, save_path)

        msg = "Save averaged top{} model at {}. Epochs: {}; mIOU: {}".format(
            len(self.top_models),
            save_path,
            epochs,
            ["{:.4f}".format(x) for x in mean_ious]
        )
        print(msg)
        self.printer.cprint(msg)

    def train(self):

        self.backup()

        best_mIOU = 0.0

        start = time.time()
        for epoch_id in range(self.cfg['TRAIN']['num_epochs']):
            print("Training step [{:3d}/{:3d}]".format(epoch_id + 1, self.cfg['TRAIN']['num_epochs']))
            self.train_epoch(epoch_id)

            if (epoch_id + 1) >= 1:
                print("Testing step [{:3d}/{:3d}]".format(epoch_id + 1, self.cfg['TRAIN']['num_epochs']))
                best_mIOU, eval_outs = self.eval(epoch_id, best_mIOU)

                if self.cfg['TRAIN']['lr_scheduler'] in REDUCE_LR_SCHEDULERS:
                    self.scheduler.step(eval_outs[1])

        self.save_averaged_topk_model()

        self.writer.close()
        end = time.gmtime(time.time() - start)
        print('Total training time is:', time.strftime("%H:%M:%S", end))

    def train_epoch(self, epoch_id):

        training_loss = 0.0
        count = 0
        iteration = 0

        self.model.train()
        print('Current learning rate: %e' % (self.opt.state_dict()['param_groups'][0]['lr']))

        for ev_rep, img, label in tqdm(self.training_loader):
            self.cur_iter = epoch_id * len(self.training_loader) + iteration
            self.scheduler.cur_iter = self.cur_iter

            ev_rep, img, label = ev_rep.type(torch.FloatTensor).to(self.cfg['DEVICE']), \
                img.type(torch.FloatTensor).to(self.cfg['DEVICE']), label.to(self.cfg['DEVICE'])

            self.opt.zero_grad()
            pred = self.model(ev_rep, img)
            pred_label = pred.argmax(dim=1)
            loss = self.criterion(pred, label)
            loss.backward()
            self.opt.step()

            if self.cfg['TRAIN']['lr_scheduler'] not in REDUCE_LR_SCHEDULERS:
                self.scheduler.step()

            count += self.cfg['TRAIN']['batch_size']
            iteration += 1
            training_loss += loss.item() * self.cfg['TRAIN']['batch_size']
            self.metrics.update_batch(pred_label, label)

        scores = self.metrics.get_metrics_summary()

        print("Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(
            training_loss * 1.0 / count,
            scores['mean_iou'],
            scores['acc']
        ))

        log_str = "[Train]  Epoch: {:d}, Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(
            epoch_id + 1,
            training_loss * 1.0 / count,
            scores['mean_iou'],
            scores['acc']
        )

        self.printer.cprint(log_str)
        self.writer.add_scalar('train/' + 'train_loss', training_loss * 1.0 / count, epoch_id + 1)
        self.writer.add_scalar('train/' + 'train_mIOU', scores['mean_iou'], epoch_id + 1)
        self.writer.add_scalar('train/' + 'train_acc', scores['acc'], epoch_id + 1)

        if (epoch_id + 1) % self.cfg['TRAIN']['save_every_n_epochs'] == 0:
            save_path = os.path.join(
                self.cfg['TRAIN']['log_dir'],
                'checkpoint_epoch_' + str((epoch_id + 1)) + '.pth'
            )
            torch.save({'epoch': epoch_id + 1, 'state_dict': self.model.state_dict()}, save_path)
            print("Save the model at {}".format(save_path))

        self.metrics.reset()

    def eval(self, epoch_id, best_mIOU):

        testing_loss = 0.0
        count = 0

        self.model.eval()

        with torch.no_grad():
            for ev_rep, img, label in tqdm(self.validation_loader):
                ev_rep, img, label = ev_rep.type(torch.FloatTensor).to(self.cfg['DEVICE']), \
                    img.type(torch.FloatTensor).to(self.cfg['DEVICE']), label.to(self.cfg['DEVICE'])

                pred = self.model(ev_rep, img)
                pred_label = pred.argmax(dim=1)
                loss = self.criterion(pred, label)

                count += self.cfg['TRAIN']['batch_size']
                testing_loss += loss.item() * self.cfg['TRAIN']['batch_size']
                self.metrics.update_batch(pred_label, label)

        scores = self.metrics.get_metrics_summary()
        eval_outs = [testing_loss * 1.0 / count, scores['mean_iou'], scores['acc']]

        print("Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(
            testing_loss * 1.0 / count,
            scores['mean_iou'],
            scores['acc']
        ))

        log_str = "[Test]   Epoch: {:d}, Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(
            epoch_id + 1,
            testing_loss * 1.0 / count,
            scores['mean_iou'],
            scores['acc']
        )

        self.printer.cprint(log_str)
        self.writer.add_scalar('test/' + 'test_loss', testing_loss * 1.0 / count, epoch_id + 1)
        self.writer.add_scalar('test/' + 'test_mIOU', scores['mean_iou'], epoch_id + 1)
        self.writer.add_scalar('test/' + 'test_acc', scores['acc'], epoch_id + 1)

        self._update_top_models(epoch_id, scores['mean_iou'])

        if scores['mean_iou'] >= best_mIOU:
            best_mIOU = scores['mean_iou']
            save_path = os.path.join(self.cfg['TRAIN']['log_dir'], 'best_model.pth')
            torch.save({'epoch': epoch_id + 1, 'state_dict': self.model.state_dict()}, save_path)

            print("Save the best model at {}".format(save_path))
            print('New best mIOU is %.4f' % best_mIOU)
            self.printer.cprint('New best mIOU is %.4f' % best_mIOU)

        if scores['mean_iou'] >= 70.00:
            torch.save(
                {'epoch': epoch_id + 1, 'state_dict': self.model.state_dict()},
                os.path.join(self.cfg['TRAIN']['log_dir'], f"{scores['mean_iou']}.pth")
            )

        self.metrics.reset()
        return best_mIOU, eval_outs

    def backup(self):
        root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        if not os.path.exists(self.cfg['TRAIN']['log_dir'] + '/' + 'Backup'):
            os.makedirs(self.cfg['TRAIN']['log_dir'] + '/' + 'Backup')

        model_saving_dir = os.path.join(self.cfg['TRAIN']['log_dir'], 'Backup')

        os.system('cp %s ' % (os.path.join(root, 'configs/DDD17.yaml')) + model_saving_dir + '/' + 'DDD17.yaml.backup')
        os.system('cp %s ' % (os.path.join(root, 'configs/DSEC_Semantic.yaml')) + model_saving_dir + '/' + 'DSEC_Semantic.yaml.backup')
        os.system('cp %s ' % (os.path.join(root, 'utils/trainer.py')) + model_saving_dir + '/' + 'trainer.py.backup')
        os.system('cp %s ' % (os.path.join(root, 'model/fuse_modules.py')) + model_saving_dir + '/' + 'fuse_modules.py.backup')
        os.system('cp %s ' % (os.path.join(root, 'model/segmentors.py')) + model_saving_dir + '/' + 'segmentors.py.backup')
        os.system('cp %s ' % (os.path.join(root, 'datasets/data_util.py')) + model_saving_dir + '/' + 'data_util.py.backup')
        os.system('cp %s ' % (os.path.join(root, 'datasets/ddd17_dataset.py')) + model_saving_dir + '/' + 'ddd17_dataset.py.backup')
        os.system('cp %s ' % (os.path.join(root, 'datasets/extract_data_tools/DSEC/sequence.py')) + model_saving_dir + '/' + 'sequence.py.backup')