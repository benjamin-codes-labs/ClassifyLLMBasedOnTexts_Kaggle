"""Generate a self-contained submission notebook from this project's sources."""
import argparse
import json
from pathlib import Path


def cell(text, markdown=False):
    result={"cell_type":"markdown" if markdown else "code","metadata":{},
            "source":text.splitlines(keepends=True)}
    if not markdown:
        result.update(execution_count=None,outputs=[])
    return result


def create(selection_path, destination):
    root=Path(__file__).resolve().parent
    selection=json.loads(Path(selection_path).read_text())
    encoder=selection["specification"]["encoder"]
    cells=[
        cell("# Name That LLM — independent Project 2 solution\n\n"
             "Attach the competition data, disable internet, and run all cells. "
             "This notebook rebuilds the selected model on all labeled rows and "
             "predicts the actual hidden test file. The selection comes from five "
             "validation folds grouped by prompt, with a separate 30-prompt holdout.\n\n"
             "This implementation was written from scratch without research-paper "
             "or competitor-notebook code. Predictions do not depend on which other "
             "passages appear in the test batch.\n\n"
             + ("**GPU input required:** attach the `checkpoints/electra-large/` folder "
                "as a private dataset, including its `manifest.json`, tokenizer, "
                "configuration, and model weights.\n" if encoder else
                "**CPU version:** no pretrained checkpoint input is needed.\n"),True),
        cell("import os, sys, json, time\n"
             "from pathlib import Path\n"
             "os.environ['HF_HUB_OFFLINE']='1'\n"
             "os.environ['TRANSFORMERS_OFFLINE']='1'\n"
             "os.environ['TOKENIZERS_PARALLELISM']='false'\n"
             "WORK=Path(os.environ.get('NTL_WORK_DIR','/kaggle/working'))\n"
             "WORK.mkdir(parents=True,exist_ok=True)\n"
             "sys.path.insert(0,str(WORK))\n"
             "if os.environ.get('NTL_DATA_DIR'):\n"
             "    DATA=Path(os.environ['NTL_DATA_DIR'])\n"
             "else:\n"
             "    candidates=[p.parent for p in Path('/kaggle/input').rglob('sample_submission.parquet')\n"
             "                if (p.parent/'train.csv').exists() and (p.parent/'test.csv').exists()]\n"
             "    if len(candidates)!=1:\n"
             "        raise RuntimeError(f'Expected one competition dataset, found {candidates}')\n"
             "    DATA=candidates[0]\n"
             "print('Data:',DATA)\n"
             "started=time.perf_counter()\n"),
    ]
    sources=["dataio.py","attribution.py"]+(["encode.py"] if encoder else [])
    if any(name.startswith("neural_") for name in selection["selected"]["weights"]):
        sources.append("neural.py")
    for name in sources:
        cells.append(cell(f"(WORK/{name!r}).write_text({(root/name).read_text()!r})\n"))
    cells.append(cell(
        "import pandas as pd\n"
        "from threadpoolctl import threadpool_limits\n"
        "from dataio import read_competition, digest_file, submit, embedding_rows, embedding_spec\n"
        "from attribution import Predictor, weighted_predictions, VERSION\n"
        f"selection=json.loads({json.dumps(json.dumps(selection))})\n"
        "spec=selection['specification']\n"
        "assert spec['version']==VERSION\n"
        "if digest_file(DATA/'train.csv')!=spec['train_sha256']:\n"
        "    raise RuntimeError('Training data changed. Revalidate before submitting.')\n"
        "train,test,labels=read_competition(DATA)\n"
        "assert labels==spec['labels']\n"
        "y=pd.Categorical(train.model,categories=labels).codes\n"
        "print('Training rows:',len(train),'Runtime test rows:',len(test))\n"
    ))
    if encoder:
        cells.append(cell(
            "from encode import encode_frame\n"
            "if os.environ.get('NTL_CHECKPOINT_DIR'):\n"
            "    checkpoint=Path(os.environ['NTL_CHECKPOINT_DIR'])\n"
            "else:\n"
            "    matches=[]\n"
            "    for p in Path('/kaggle/input').rglob('manifest.json'):\n"
            "        try:\n"
            "            manifest=json.loads(p.read_text())\n"
            "        except (ValueError,UnicodeError):\n"
            "            continue\n"
            "        if manifest==spec['encoder']['checkpoint']:\n"
            "            matches.append(p.parent)\n"
            "    if len(matches)!=1:\n"
            "        raise RuntimeError('Attach the exact ELECTRA checkpoint folder recorded in the selection')\n"
            "    checkpoint=matches[0]\n"
            "os.environ['NTL_CHECKPOINT_DIR']=str(checkpoint)\n"
            "frame=pd.concat([train[['id','text']],test[['id','text']]],ignore_index=True)\n"
            "cache=WORK/'encoder_vectors.npz'\n"
            "encode_frame(frame,checkpoint,cache)\n"
            "import torch\n"
            "if torch.cuda.is_available():\n"
            "    torch.cuda.empty_cache()\n"
            "elif torch.backends.mps.is_available():\n"
            "    torch.mps.empty_cache()\n"
            "if embedding_spec(cache)!=spec['encoder']:\n"
            "    raise RuntimeError('Wrong encoder revision or configuration')\n"
            "training_vectors=embedding_rows(cache,train)\n"
            "test_vectors=embedding_rows(cache,test)\n"
        ))
    else:
        cells.append(cell("training_vectors=test_vectors=None\n"))
    cells.append(cell(
        "weights=selection['selected']['weights']\n"
        "with threadpool_limits(limits=4):\n"
        "    predictor=Predictor(labels,selected=weights).fit(train.text.tolist(),y,training_vectors,train.prompt_id.to_numpy())\n"
        "    scores=weighted_predictions(predictor.predict_members(test.text.tolist(),test_vectors),weights)\n"
        "submission=submit(test,scores,labels,WORK/'submission.parquet')\n"
        "assert submission.id.tolist()==test.id.tolist()\n"
        "print(submission.head())\n"
        "print(f'Wrote {len(submission)} predictions in {(time.perf_counter()-started)/60:.1f} minutes')\n"
    ))
    for index,c in enumerate(cells):
        c["id"]=f"project2-{index:03}"
    notebook={"nbformat":4,"nbformat_minor":5,"cells":cells,"metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.11"},
        "kaggle":{"isInternetEnabled":False,"isGpuEnabled":bool(encoder),"language":"python"},
    }}
    Path(destination).write_text(json.dumps(notebook,indent=1))
    print("Created",destination)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--selection",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    create(args.selection,args.output)
