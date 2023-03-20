from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score, jaccard_score, f1_score

import os
import re

from collections import namedtuple, Counter
Metrics = namedtuple("Metrics", "ap ba jaccard dice label")

class MetricCalculator:
    
    GT_THRESHOLD = 0.5
    PRED_THRESHOLD = 0.5
    
    def __init__(self,path_pred):
        self.path_pred=path_pred
        pass

    def process_image(self, pred_fn, gt_fn, label=None, verbose=False):
        pred_img = Image.open(pred_fn).convert('L')
        gt_img = Image.open(gt_fn).convert('L')

        pred_array = 1 - np.asarray(pred_img) / 255
        gt_array = 1 - np.asarray(gt_img) / 255
        # 1 - x : assumes background is x=1

        gt_uniq = np.unique(gt_array)
        gt_nuniq = len(gt_uniq)
        if gt_nuniq != 2:
            print(f'Warning: Ground truth mask contains {gt_nuniq} values unique:', *gt_uniq)
            print('Using threshold', self.GT_THRESHOLD)

        gt_mask = gt_array > self.GT_THRESHOLD

        gt_mask_flat = gt_mask.ravel()
        pred_array_flat = pred_array.ravel()

        pred_array_flat_thresh = pred_array_flat > self.PRED_THRESHOLD

        # calculate metrics
        AP = average_precision_score(gt_mask_flat, pred_array_flat)

        tpr = (gt_mask_flat * pred_array_flat).sum() / gt_mask_flat.sum()
        tnr = ((1 - gt_mask_flat) * (1 - pred_array_flat)).sum() / (1 - gt_mask_flat).sum()

        balanced_acc = (tpr + tnr) / 2

        jscore = jaccard_score(gt_mask_flat, pred_array_flat_thresh)

        dice = f1_score(gt_mask_flat, pred_array_flat_thresh)

        print('AP (average precision):\t', AP)
        print('Balanced accuracy:\t', balanced_acc)
        print('Jaccard score (IoU):\t', jscore, f'(Threshold: {self.PRED_THRESHOLD})')
        print('Dice score (F1):\t', dice, f'(Threshold: {self.PRED_THRESHOLD})')
        
        metrics = Metrics(AP, balanced_acc, jscore, dice, label)

        # if verbose:
        with open(os.path.join(self.path_pred, pred_fn.split('.')[0] + '_metric.txt'),'w')as file:
            file.write(f'Metrics for single image (GT: {gt_fn}; preds: {pred_fn})\n')
            file.write(f'\tAP (average precision):\t {metrics.ap}\n')
            file.write(f'\tBalanced accuracy:\ {metrics.ba}\n')
            file.write(f'\tJaccard score (IoU):\t {metrics.jaccard}, Threshold: {self.PRED_THRESHOLD}\n')
            file.write(f'\tDice score (F1):\t {metrics.dice}, Threshold: {self.PRED_THRESHOLD}')
            file.close()
        return metrics
        
    
    def process_all_images(self, pred_fns, gt_fns,labels):
        assert len(pred_fns) == len(gt_fns) == len(labels), 'Mismatched number of filenames and/or labels'
        all_metrics = [
            self.process_image(pred_fn, gt_fn, label)
            for pred_fn, gt_fn, label in zip(pred_fns, gt_fns,labels)
        ]
        
        with open(self.path_pred + '/all_category_metric.txt','w')as f:
            f.write("all_category_metric :\n")
            f.write('\tAP (average precision):\t' + str(np.mean([m.ap for m in all_metrics])) + '\n')
            f.write('\tBalanced accuracy:\t' + str(np.mean([m.ba for m in all_metrics])) + '\n')
            f.write('\tJaccard score (IoU):\t' + str(
                np.mean([m.jaccard for m in all_metrics])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\tDice score (F1):\t' + str(
                np.mean([m.dice for m in all_metrics])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\n')

        def macro_average(metric_name):
            values = [
                np.mean([getattr(m, metric_name) for m in all_metrics])
            ]
            return np.mean(values)
        return {
            k: macro_average(k) for k in ['ap', 'ba', 'jaccard', 'dice']
        }


if __name__ == "__main__":

    images=[]
    images_gt = []
    scores = []
    scores_gt = []
    mask_gt = []
    
    gt_path = "/storage/chendudai/data/manually_gt_masks_0_1/window"
    eval_path = "/storage/hanibezalel/Ha-NeRF/eval/results/phototourism/0_1_withSemantics_100_com_gt/eval/results/phototourism/0_1_withSemantics_100_com_gt"
    #ts_list = [17, 23, 29, 89, 131, 633]


    gt_idx = []
    gt_files = []
    pred_idx = []
    for file in os.listdir(gt_path):
        if re.findall('[0-9]+',file):
            #if int(re.findall('\d+',file)[0]) in ts_list:
            gt_idx.append(int(re.findall('[0-9]+',file)[0]))
            gt_files.append(file)
       
    for file in os.listdir(eval_path):
        if file.endswith(".png"):
            int_idx = int(re.findall('\d+',file)[0])
            if int(re.findall('\d+',file)[0]) in gt_idx:
                #if re.match(file,'[0-9][0-9][0-9]'):
                #    images[re.find('[0-9][0-9][0-9]',file)] = os.path.join(path, file)
                #elif re.match(file,'[0-9][0-9][0-9]_s_gt'):
                #    images_gt[re.find('[0-9][0-9][0-9]',file)] = os.path.join(path, file)
                #elif re.match(file,'[0-9][0-9][0-9]_s_f'):
                if re.findall('s_f',file):
                    scores.append(os.path.join(eval_path, file))
                    pred_idx.append(int(re.findall('[0-9]+',file)[0]))
                    mask_gt.append(os.path.join(gt_path, gt_files[gt_idx.index(int_idx)]))
                #elif re.match(file,'[0-9][0-9][0-9]_s_gt'):
                #    scores_gt[re.find('[0-9][0-9][0-9]',file)] = os.path.join(path, file)
            else:
                continue
            
    #rmv_idx = list(set(gt_idx) - set(pred_idx))
    #for file in os.listdir(gt_path):
    #    if re.findall('[0-9]+',file):
    #        if int(re.findall('\d+',file)[0]) in rmv_idx:
    #            mask_gt.remove(os.path.join(gt_path, file))
        
                    

    labels = ['window'] * len(mask_gt)
    calculator = MetricCalculator(eval_path)
    calculator.process_all_images(scores,mask_gt,labels)