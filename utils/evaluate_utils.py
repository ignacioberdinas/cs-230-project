import time
import torch
from utils.coco_utils import get_coco_api_from_dataset, CocoEvaluator


@torch.no_grad()
def evaluate(model, data_loader, device, mAP_list=None):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    print('Running eval')

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in data_loader:
        image = list(img.to(device) for img in image)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        for o in outputs:
            o['boxes'] = torch.tensor([o['boxes'][0].cpu().numpy()])
            o['labels'] =  torch.tensor([o['labels'][0].item()])
            o['scores'] =  torch.tensor([o['scores'][0].item()])

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        #print('Evaluator time ' + str(evaluator_time))

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    print_txt = coco_evaluator.coco_eval[iou_types[0]].stats
    print(print_txt)
    coco_mAP = print_txt[0]
    voc_mAP = print_txt[1]
    print("coco_mAP",coco_mAP)
    print("voc_mAP",voc_mAP)
    if isinstance(mAP_list, list):
        mAP_list.append(voc_mAP)

    return coco_evaluator, voc_mAP