"""Evaluate the transferred, detection-only SAM checkpoint on Apple MPS."""
import argparse, json, time, sys, gc, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor'))
import torch
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_cxcywh_to_xyxy

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint', required=True, help='Path to checkpoint.pt')
 p.add_argument('--data-root', required=True, help='Directory containing val.json and val_imgs/')
 p.add_argument('--limit',type=int,default=8, help='Validation images; 0 evaluates all')
 p.add_argument('--output',default='results/mps', help='Directory for metrics and visualizations')
 a=p.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 assert torch.backends.mps.is_available()
 torch.set_num_threads(6)
 start=time.perf_counter()
 print('Building model on CPU',flush=True)
 model=build_sam3_image_model(device='cpu',load_from_HF=False,enable_segmentation=False)
 print('Loading checkpoint (memory mapped, weights_only)',flush=True)
 ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
 weights=ckpt.get('model',ckpt)
 if any(k.startswith('detector.') for k in weights): weights={k.removeprefix('detector.'):v for k,v in weights.items() if k.startswith('detector.')}
 status=model.load_state_dict(weights,strict=True)
 metadata={'torch':torch.__version__,'device':'mps','dtype':'float32','load_status':str(status),'checkpoint_keys':list(ckpt),'epoch':ckpt.get('epoch'),'resolution':1008}
 del weights,ckpt; gc.collect()
 # Preserve learned parameters; use upstream real-valued rotary arithmetic on MPS.
 for mod in model.modules():
  if hasattr(mod,'freqs_cis') and isinstance(mod.freqs_cis,torch.Tensor) and mod.freqs_cis.is_complex():
   freq=mod.freqs_cis
   mod.use_rope_real=True
   mod.register_buffer('freqs_cis_real',freq.real.contiguous(),persistent=False)
   mod.register_buffer('freqs_cis_imag',freq.imag.contiguous(),persistent=False)
   mod._buffers.pop('freqs_cis',None)
   mod.freqs_cis=freq # CPU-only reference; real buffers move to MPS
 model=model.to('mps').eval()
 processor=Sam3Processor(model,device='mps',confidence_threshold=0.0)
 root=Path(a.data_root); gt=json.loads((root/'val.json').read_text())
 images=sorted(gt['images'],key=lambda x:x['id'])
 if a.limit: images=sorted(random.Random(42).sample(images,min(a.limit,len(images))),key=lambda x:x['id'])
 predictions=[]; timings=[]
 metadata['load_seconds']=time.perf_counter()-start
 print('Model ready',metadata['load_seconds'],flush=True)
 with torch.inference_mode():
  for img in images:
   t=time.perf_counter(); pil=Image.open(root/'val_imgs'/Path(img['file_name']).name).convert('RGB')
   state=processor.set_image(pil); torch.mps.synchronize(); enc=time.perf_counter()-t
   image_features = {key: state['backbone_out'][key] for key in ('backbone_fpn', 'vision_features', 'vision_pos_enc')}
   for cat in gt['categories']:
    # `forward_text` returns a dict that shares a `backbone_fpn` key in this
    # training checkpoint. Keep the image features produced by `set_image`.
    text_out = model.backbone.forward_text([cat['name']], device='mps')
    state['backbone_out'].update(image_features)
    for key, value in text_out.items():
     if key not in {'backbone_fpn', 'vision_features', 'vision_pos_enc'}:
      state['backbone_out'][key] = value
    raw=model.forward_grounding(backbone_out=state['backbone_out'],find_input=processor.find_stage,geometric_prompt=model._get_dummy_prompt(),find_target=None)
    scores=(raw['pred_logits'].sigmoid()*raw['presence_logit_dec'].sigmoid().unsqueeze(1)).flatten().cpu()
    boxes=box_cxcywh_to_xyxy(raw['pred_boxes']).reshape(-1,4).cpu()*torch.tensor([pil.width,pil.height,pil.width,pil.height])
    for box,score in zip(boxes.tolist(),scores.tolist()):
     x,y,x2,y2=box
     predictions.append(dict(image_id=img['id'],category_id=cat['id'],bbox=[x,y,x2-x,y2-y],score=score))
   torch.mps.synchronize(); elapsed=time.perf_counter()-t
   timings.append({'image_id':img['id'],'encoder_seconds':enc,'total_seconds':elapsed})
   print(f"Image {img['id']}: {elapsed:.2f}s, encoder {enc:.2f}s",flush=True)
   canvas=Image.new('RGB',(pil.width*2,pil.height+30),'white');canvas.paste(pil,(0,30));canvas.paste(pil,(pil.width,30)); draw=ImageDraw.Draw(canvas);draw.text((5,5),'Ground truth',fill='black');draw.text((pil.width+5,5),'MPS predictions (score >= 0.35)',fill='black')
   cats={c['id']:c['name'] for c in gt['categories']}
   for items,offset in [(gt['annotations'],0),(predictions,pil.width)]:
    for d in items:
     if d['image_id']!=img['id'] or d.get('score',1)<.35: continue
     x,y,w,h=d['bbox'];color=f"hsl({d['category_id']*137%360},85%,45%)";draw.rectangle((x+offset,y+30,x+w+offset,y+h+30),outline=color,width=2);draw.text((x+offset,y+30),cats[d['category_id']]+(f" {d['score']:.2f}" if 'score' in d else ''),fill=color,stroke_width=1,stroke_fill='white')
   canvas.save(out/f"image_{img['id']:04d}.jpg")
   (out/'predictions.json').write_text(json.dumps(predictions)); (out/'timings.json').write_text(json.dumps(timings,indent=2))
   del state,raw; torch.mps.empty_cache()
 metadata['image_ids']=[i['id'] for i in images];metadata['mps_allocated_bytes']=torch.mps.current_allocated_memory(); (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
 from pycocotools.coco import COCO
 from pycocotools.cocoeval import COCOeval
 coco=COCO(str(root/'val.json')); ev=COCOeval(coco,coco.loadRes(predictions),'bbox');ev.params.imgIds=metadata['image_ids'];ev.evaluate();ev.accumulate();ev.summarize()
 metrics={'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP75':float(ev.stats[2]),'images':len(images),'per_class':{}}
 for k,c in enumerate(gt['categories']):
  q=ev.eval['precision'][:,:,k,0,2];q50=q[0]
  valid=q[q>-1]; valid50=q50[q50>-1]
  metrics['per_class'][c['name']]={
   'AP': float(valid.mean()) if len(valid) else None,
   'AP50': float(valid50.mean()) if len(valid50) else None,
  }
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
