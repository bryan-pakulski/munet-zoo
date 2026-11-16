"""End-to-end RT-DETRv2 training, resumable state and inference export."""
from collections import OrderedDict
import copy
import numpy as np
from munet import compile, optim, nn
from munet.checkpoint import save_state,load_state
from . import RTDETRv2, RTDETRCriterionv2, pad_targets, prepare_denoising


def rtdetrv2_adamw(model, lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4):
    """AdamW groups: the backbone trains at backbone_lr; biases and normalization parameters get no weight decay."""
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        groups[name.startswith('backbone.'), len(p.shape) > 1].append(p)
    return optim.AdamW([{'params': params, 'lr': backbone_lr if backbone else lr, **({} if decay else {'weight_decay': 0})}
                        for (backbone, decay), params in groups.items() if params], weight_decay=weight_decay)


class DetectorTrainer:
    """Fresh denoising per batch; forward, loss, backward, AdamW and EMA in one replay.

    Each cached specialization fixes image/batch/target-bucket dimensions. Switching
    specializations synchronizes state through the host; steady-shape replay keeps
    parameters, gradients, moments and EMA on the selected device.
    """
    def __init__(self,model,*,device='vulkan',optimizer=None,criterion=None,target_slots=None,
                 max_norm=.1,ema=True,seed=0,milestones=(1000,),gamma=.1,max_cached_shapes=2):
        if target_slots is not None and not 1<=target_slots<=min(model.num_queries,512): raise ValueError('target_slots must fit the matcher/query count')
        if max_cached_shapes<1: raise ValueError('at least one cached shape is required')
        self.model=model.train();self.device=device
        self.optimizer=optimizer or rtdetrv2_adamw(model);self.criterion=criterion or RTDETRCriterionv2(model.num_classes)
        self.target_slots,self.max_norm=target_slots,max_norm;self.max_cached_shapes=max_cached_shapes
        self.ema=optim.ModelEMA(model) if ema else None
        self.scheduler=optim.MultiStepLR(self.optimizer,milestones,gamma)
        self.rng=np.random.default_rng(seed);self.steps=0;self.epoch=0;self.programs=OrderedDict()

    def _program(self,dn):
        def step(images,labels,boxes,valid,*noise):
            self.optimizer.zero_grad()
            payload=None if dn is None else {'labels':noise[0],'boxes_unact':noise[1],'attn_mask':noise[2],
                                              'num_group':dn['num_group'],'max_gt':dn['max_gt']}
            losses=self.criterion(self.model(images,denoising=payload),{'labels':labels,'boxes':boxes,'valid':valid})
            total=self.criterion.total(losses);total.backward()
            norm=optim.clip_grad_norm_(self.model.parameters(),self.max_norm)
            self.optimizer.step()
            if self.ema: self.ema.update()
            return {'loss':total,'grad_norm':norm,**losses}
        return compile(step,device=self.device)

    @staticmethod
    def _detach_owners(program):
        for p,v in program._parameters.items():
            if p._owner is not None and p._owner[0] is program:
                p._array=program._plan.read(v);p._owner=None

    def step(self,images,targets):
        self.model.train();images=np.asarray(images)
        if images.dtype!=np.float32 or len(images)!=len(targets) or not np.isfinite(images).all(): raise ValueError('expected finite float32 images and one target per image')
        padded=pad_targets(targets,self.model.num_classes,self.target_slots)
        if padded['valid'].shape[1]>self.model.num_queries: raise ValueError('more ground-truth slots than queries; increase num_queries or reduce target bucket')
        dn=prepare_denoising(padded,self.model.num_classes,self.model.num_queries,
                            num_denoising=self.model.config['num_denoising'],rng=self.rng)
        arrays=[images,padded['labels'],padded['boxes'],padded['valid']]
        if dn is not None: arrays.extend([dn['labels'],dn['boxes_unact'],dn['attn_mask']])
        key=(tuple(a.shape for a in arrays),None if dn is None else (dn['max_gt'],dn['num_group']))
        if key not in self.programs:
            if len(self.programs)>=self.max_cached_shapes:
                _,old=self.programs.popitem(last=False);self._detach_owners(old)
            self.programs[key]=self._program(dn)
        program=self.programs[key];self.programs.move_to_end(key)
        result=program(*arrays);self.steps+=1
        return {k:v.item() for k,v in result.items()}

    def finish_epoch(self):
        self.epoch+=1;self.scheduler.step(self.epoch)

    def state_dict(self):
        return {'config':self.model.config,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                'ema':None if self.ema is None else self.ema.state_dict(),'scheduler':self.scheduler.state_dict(),
                'rng':copy.deepcopy(self.rng.bit_generator.state),'steps':self.steps,'epoch':self.epoch,
                'target_slots':self.target_slots,'max_norm':self.max_norm,
                'trainable':[name for name,p in self.model.named_parameters() if p.requires_grad]}

    def save(self,path): save_state(self.state_dict(),path)

    def load(self,path):
        state=load_state(path)
        if state['config']!=self.model.config or state['target_slots']!=self.target_slots or state['max_norm']!=self.max_norm: raise ValueError('training configuration mismatch')
        if state['trainable']!=[n for n,p in self.model.named_parameters() if p.requires_grad]: raise ValueError('trainable parameter selection mismatch')
        if (state['ema'] is None)!=(self.ema is None): raise ValueError('EMA configuration mismatch')
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
        if self.ema: self.ema.load_state_dict(state['ema'])
        self.scheduler.load_state_dict(state['scheduler']);self.rng.bit_generator.state=state['rng']
        self.steps,self.epoch=state['steps'],state['epoch']
        return self

    def inference_model(self,use_ema=True):
        model=RTDETRv2(**self.model.config).eval()
        if use_ema and self.ema: self.ema.copy_to(model)
        else: model.load_state_dict(self.model.state_dict())
        return model

    def export(self,path,example_images,*,onnx_path=None,use_ema=True):
        from munet.serialization import save
        model=self.inference_model(use_ema);program=compile(model,device=self.device);program(example_images)
        save(program,path)
        if onnx_path is not None:
            from munet.interop import to_onnx
            to_onnx(program,onnx_path)
        return program


def load_torch_checkpoint(model, path, *, prefer_ema=True):
    """Load an official RT-DETRv2 PyTorch checkpoint into a model built with the matching ``presnet`` backbone.

    Nothing is downloaded. PyTorch is needed only to read the file.
    """
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if prefer_ema and isinstance(state, dict) and state.get('ema') is not None:
        state = state['ema']; state = state.get('module', state)
    elif isinstance(state, dict) and 'model' in state: state = state['model']
    model.load_state_dict(upstream_state(state))
    return model


def upstream_state(state):
    """Drop the constants the PyTorch implementation stores as buffers; here they are computed, not loaded."""
    constant = ('.num_points_scale', 'decoder.anchors', 'decoder.valid_mask')
    return {name: value for name, value in state.items() if not name.endswith(constant)}
