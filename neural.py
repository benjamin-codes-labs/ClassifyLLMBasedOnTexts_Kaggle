"""A compact fine-tuning experiment with fixed settings and grouped validation."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from dataio import SEED, digest_file, read_competition, split_prompts

RECIPE = {"epochs": 3, "last_layers": 4, "batch_size": 6, "accumulate": 4,
          "encoder_lr": 1e-5, "head_lr": 3e-4, "weight_decay": .02,
          "label_smoothing": .03, "max_length": 512, "version": "neural-1"}


def checkpoint_directory():
    return Path(os.environ.get("NTL_CHECKPOINT_DIR","checkpoints/electra-large"))


def device_name():
    import torch
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")


def network(path, classes, statistics_width):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=AutoModel.from_pretrained(path,local_files_only=True)
            for p in self.encoder.parameters():
                p.requires_grad=False
            for layer in self.encoder.encoder.layer[-RECIPE["last_layers"]:]:
                for p in layer.parameters():
                    p.requires_grad=True
            width=self.encoder.config.hidden_size*2+statistics_width
            self.head=nn.Sequential(nn.LayerNorm(width),nn.Dropout(.2),
                                    nn.Linear(width,256),nn.GELU(),
                                    nn.Dropout(.2),nn.Linear(256,classes))

        def forward(self,input_ids,attention_mask,statistics,**kwargs):
            h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(h.dtype)
            mean=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
            return self.head(torch.cat([mean,h[:,0],statistics],dim=1))
    return Network()


class NeuralModel:
    def __init__(self,classes,epochs):
        self.classes=int(classes)
        self.epochs=int(epochs)
        self._network=None
        self._tokenizer=None

    def __getstate__(self):
        # Save only learned tensors and configuration, not device-bound objects.
        state=self.__dict__.copy()
        state["_network"]=None
        state["_tokenizer"]=None
        return state

    def _initialize(self,stats_width):
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        self._device=device_name()
        path=checkpoint_directory()
        manifest=json.loads((path/"manifest.json").read_text())
        if hasattr(self,"manifest") and self.manifest!=manifest:
            raise ValueError("Neural checkpoint revision mismatch")
        self.manifest=manifest
        self._tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self._network=network(path,self.classes,stats_width).to(self._device)

    def _tokenize(self,texts):
        return self._tokenizer(texts,truncation=True,max_length=RECIPE["max_length"],
                               padding=False,return_token_type_ids=False)["input_ids"]

    def fit(self,texts,y,validation=None,callback=None):
        import torch
        from attribution import numeric_features
        self.scaler=StandardScaler().fit(numeric_features(texts))
        statistics=np.clip(self.scaler.transform(numeric_features(texts)),-8,8).astype(np.float32)
        self._initialize(statistics.shape[1])
        tokens=self._tokenize(texts)
        encoder_params=[p for p in self._network.encoder.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW([
            {"params":encoder_params,"lr":RECIPE["encoder_lr"]},
            {"params":self._network.head.parameters(),"lr":RECIPE["head_lr"]},
        ],weight_decay=RECIPE["weight_decay"])
        updates_per_epoch=math.ceil(math.ceil(len(texts)/RECIPE["batch_size"])/RECIPE["accumulate"])
        # An epoch-2 checkpoint must use the same schedule when refitted.
        total_updates=updates_per_epoch*RECIPE["epochs"]
        warmup=max(1,int(.1*total_updates))
        def factor(step):
            return min((step+1)/warmup,max(0,(total_updates-step)/max(1,total_updates-warmup)))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=RECIPE["label_smoothing"])
        started=time.perf_counter()
        for epoch in range(self.epochs):
            rng=np.random.default_rng(SEED+epoch)
            order=rng.permutation(len(texts))
            batches=[]
            for start in range(0,len(order),120):
                bucket=sorted(order[start:start+120],key=lambda i:len(tokens[i]))
                batches.extend([bucket[i:i+RECIPE["batch_size"]] for i in range(0,len(bucket),RECIPE["batch_size"])])
            rng.shuffle(batches)
            self._network.train()
            # Frozen lower-layer dropout is disabled to stabilize representations.
            self._network.encoder.embeddings.eval()
            for layer in self._network.encoder.encoder.layer[:-RECIPE["last_layers"]]:
                layer.eval()
            optimizer.zero_grad(set_to_none=True)
            loss_sum=0
            for step,batch in enumerate(batches):
                inputs=self._tokenizer.pad({"input_ids":[tokens[i] for i in batch]},
                                           padding=True,return_tensors="pt").to(self._device)
                z=torch.tensor(statistics[batch],device=self._device)
                target=torch.tensor(y[batch],dtype=torch.long,device=self._device)
                output=self._network(**inputs,statistics=z)
                loss=loss_fn(output,target)
                loss_sum+=float(loss.detach().cpu())*len(batch)
                # Normalize the final partial accumulation group correctly.
                group_start=(step//RECIPE["accumulate"])*RECIPE["accumulate"]
                group=batches[group_start:group_start+RECIPE["accumulate"]]
                (loss*len(batch)/sum(map(len,group))).backward()
                if (step+1)%RECIPE["accumulate"]==0 or step+1==len(batches):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self._network.parameters() if p.requires_grad],1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            print(f"  neural epoch {epoch+1}/{self.epochs}: loss={loss_sum/len(texts):.4f}; "
                  f"elapsed={time.perf_counter()-started:.0f}s",flush=True)
            if validation is not None:
                p=self.predict_proba(validation)
                if callback is not None:
                    callback(epoch+1,p)
        self.parameters={name:p.detach().cpu().numpy().copy()
                         for name,p in self._network.named_parameters() if p.requires_grad}
        self.recipe=RECIPE.copy()
        return self

    def predict_proba(self,texts):
        import torch
        from attribution import numeric_features
        numbers=np.clip(self.scaler.transform(numeric_features(texts)),-8,8).astype(np.float32)
        if self._network is None:
            if self.recipe!=RECIPE:
                raise ValueError("Neural recipe changed after training")
            self._initialize(numbers.shape[1])
            wanted={name for name,p in self._network.named_parameters() if p.requires_grad}
            if wanted!=set(self.parameters):
                raise ValueError("Neural parameter schema mismatch")
            self._network.load_state_dict({k:torch.from_numpy(v) for k,v in self.parameters.items()},strict=False)
        self._network.eval()
        tokens=self._tokenize(texts)
        order=sorted(range(len(tokens)),key=lambda i:len(tokens[i]))
        result=np.empty((len(texts),self.classes),dtype=np.float32)
        with torch.inference_mode():
            for start in range(0,len(order),RECIPE["batch_size"]):
                batch=order[start:start+RECIPE["batch_size"]]
                inputs=self._tokenizer.pad({"input_ids":[tokens[i] for i in batch]},
                                           padding=True,return_tensors="pt").to(self._device)
                z=torch.tensor(numbers[batch],device=self._device)
                result[batch]=self._network(**inputs,statistics=z).softmax(1).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite neural predictions")
        return result


def cross_validate(data,output):
    import torch
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    train,_,labels=read_competition(data)
    splits=split_prompts(train,output/"splits.csv")
    target=pd.Categorical(train.model,categories=labels).codes
    metadata={"recipe":RECIPE,"train_sha256":digest_file(Path(data)/"train.csv"),
              "checkpoint":json.loads((checkpoint_directory()/"manifest.json").read_text())}
    if (output/"specification.json").exists():
        if json.loads((output/"specification.json").read_text())!=metadata:
            raise ValueError("Neural output directory has a different recipe")
    (output/"specification.json").write_text(json.dumps(metadata,indent=2))
    oof={f"neural_e{e}":np.zeros((len(train),len(labels)),dtype=np.float32)
         for e in range(1,RECIPE["epochs"]+1)}
    for fold in range(5):
        tr=np.flatnonzero((splits.fold>=0)&(splits.fold!=fold))
        va=np.flatnonzero(splits.fold==fold)
        cache=output/f"fold_{fold}.npz"
        if cache.exists():
            with np.load(cache,allow_pickle=False) as f:
                if not np.array_equal(f["rows"],va):
                    raise ValueError("Stale neural fold")
                for name in oof:
                    oof[name][va]=f[name]
            continue
        print(f"NEURAL fold {fold}: {len(tr)} train / {len(va)} validation",flush=True)
        preds={}
        def collect(epoch,p):
            preds[f"neural_e{epoch}"]=p
            print(f"  epoch {epoch}: validation balanced accuracy "
                  f"{balanced_accuracy_score(target[va],p.argmax(1)):.4f}",flush=True)
        model=NeuralModel(len(labels),RECIPE["epochs"]).fit(
            train.text.iloc[tr].tolist(),target[tr],validation=train.text.iloc[va].tolist(),callback=collect)
        np.savez_compressed(cache,rows=va,**preds)
        for name,p in preds.items():
            oof[name][va]=p
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    rows=np.flatnonzero(splits.fold>=0)
    np.savez_compressed(output/"oof.npz",rows=rows,y=target[rows],folds=splits.fold.to_numpy()[rows],
                        **{name:p[rows] for name,p in oof.items()})
    print("Neural CV:",{name:balanced_accuracy_score(target[rows],p[rows].argmax(1))
                        for name,p in oof.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="Data")
    parser.add_argument("--output",default="runs/neural")
    args=parser.parse_args()
    cross_validate(args.data,args.output)
