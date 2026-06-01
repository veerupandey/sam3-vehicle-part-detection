"""COCO box metrics with explicit dataset/prediction paths and correct AP axes."""
import argparse,json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
 p=argparse.ArgumentParser();p.add_argument('--gt',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',default='results/evaluation.json');p.add_argument('--image-ids',type=int,nargs='+');a=p.parse_args()
 gt=COCO(a.gt);pred=json.load(open(a.predictions)); ev=COCOeval(gt,gt.loadRes(pred),'bbox')
 if a.image_ids:ev.params.imgIds=a.image_ids
 ev.evaluate();ev.accumulate();ev.summarize()
 result={'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP75':float(ev.stats[2]),'images':len(ev.params.imgIds),'per_class':{}}
 for k,cid in enumerate(ev.params.catIds):
  c=gt.cats[cid];q=ev.eval['precision'][:,:,k,0,2];q50=q[0]
  ap=float(q[q>-1].mean()) if (q>-1).any() else None;ap50=float(q50[q50>-1].mean()) if (q50>-1).any() else None
  result['per_class'][c['name']]={'AP':ap,'AP50':ap50};print(c['name'],result['per_class'][c['name']])
 with open(a.output,'w') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
