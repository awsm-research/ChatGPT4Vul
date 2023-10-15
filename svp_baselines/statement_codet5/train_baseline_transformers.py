from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_constant_schedule, RobertaTokenizerFast, T5EncoderModel, get_linear_schedule_with_warmup, RobertaModel
from tqdm import tqdm
from codet5_model_multi_task import Model

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd


logger = logging.getLogger(__name__)
global vul_training_samples
vul_training_samples = 0


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 statement_mask,
                 labels,
                 func_labels,
                 num_statements):
        self.input_ids = input_ids
        self.statement_mask = statement_mask
        self.labels = labels
        self.func_labels = func_labels
        self.num_statements = num_statements
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "val":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        
        df_all = pd.read_csv(file_path)
        
        df_vul = df_all[df_all["function_label"]==1].reset_index(drop=True)

        df_non_vul = df_all[df_all["function_label"]==0].reset_index(drop=True)
        
        df = pd.concat((df_vul, df_non_vul))
        df = df.sample(frac=1).reset_index(drop=True)
        
        patterns = df["vul_patterns"].tolist()
        labels = df["statement_label"].tolist()
        source = df["func_before"].tolist()
        
        print("\n*******\n", f"total non-vul funcs in {file_type} data: {len(df_non_vul)}")
        print(f"total vul funcs in {file_type} data: {len(df_vul)}", "\n*******\n")

        
        if file_type == "train":
            global vul_training_samples
            vul_training_samples = len(patterns)
        
        for i in tqdm(range(len(source))):
            self.examples.append(convert_examples_to_features(source[i], patterns[i], labels[i], tokenizer, args, file_type))
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("statement_mask: {}".format(' '.join(map(str, example.statement_mask))))
                logger.info(f"labels: {example.labels}")
                logger.info(f"num_statements: {example.num_statements}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].statement_mask), torch.tensor(self.examples[i].labels).float(), torch.tensor(self.examples[i].func_labels), torch.tensor(self.examples[i].num_statements)

def convert_examples_to_features(source, pattern, labels, tokenizer, args, data_split):
    labels = labels.strip("[").strip("]")
    labels = labels.split(",")
    labels = [int(l.strip()) for l in labels]
    assert len(labels) == args.num_labels
    
    source = source.split("\n")
    source = source[:args.num_labels]
    padding_statement = [tokenizer.pad_token_id for _ in range(20)]
    num_statements = len(source)
    input_ids = []
    for stat in source:
        ids_ = tokenizer.encode(str(stat),
                                truncation=True,
                                max_length=20,
                                padding='max_length',
                                add_special_tokens=False)
        input_ids.append(ids_)
    if len(input_ids) < args.num_labels:
        for _ in range(args.num_labels-len(input_ids)):
            input_ids.append(padding_statement)
    
    statement_mask = []
    for statement in input_ids:
        if statement == padding_statement:
            statement_mask.append(0)
        else:
            statement_mask.append(1)
    
    if 1 in labels:
        func_labels = 1
    else:
        func_labels = 0
    statement_mask += [1]
    
    if data_split == "train":
        if 1 not in labels:
            ids_ = tokenizer.encode(str(pattern),
                                        truncation=True,
                                        max_length=20,
                                        padding='max_length',
                                        add_special_tokens=False)
            pattern_ids = [ids_]
            for _ in range(11):
                pattern_ids.append(padding_statement)
            input_ids = input_ids + pattern_ids
        else:
            pattern = pattern.split("<SPLIT>")
            pattern = [p for p in pattern if p != ""]
            pattern_ids = []
            for pat in pattern:
                ids_ = tokenizer.encode(str(pat),
                                        truncation=True,
                                        max_length=20,
                                        padding='max_length',
                                        add_special_tokens=False)
                pattern_ids.append(ids_)
            pattern_ids = pattern_ids[:12]
            # 12 patterns - 90% of Qt.
            if len(pattern_ids) < 12:
                for _ in range(12-len(pattern_ids)):
                    pattern_ids.append(padding_statement)
            input_ids = input_ids + pattern_ids
        assert len(input_ids) == args.num_labels + 12
    else:
        assert len(input_ids) == args.num_labels
    
    return InputFeatures(input_ids, statement_mask, labels, func_labels, num_statements)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = get_constant_schedule(optimizer)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.epochs*0.1, num_training_steps=len(train_dataloader)*args.epochs)


    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, statement_mask, labels, func_labels, num_statements) = [x.to(args.device) for x in batch]
            global vul_training_samples
            model.train()
            statement_loss, func_loss = model(input_ids_with_pattern=input_ids,
                         statement_mask=statement_mask,
                         labels=labels,
                         total_training_samples=vul_training_samples,
                         func_labels=func_labels,
                         step=step)
            loss = 0.5 * statement_loss + 0.5 * func_loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_f1 = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        logger.info("  "+"*"*20)  
                        logger.info("  Best F1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)               
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (input_ids, statement_mask, labels, func_labels, num_statements) = [x.to(args.device) for x in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                        statement_mask=statement_mask)
            preds = torch.where(probs>0.5, 1, 0).tolist()
            
            func_preds = torch.argmax(func_probs, dim=-1).tolist()
            
            for indx in range(len(preds)):
                sample = preds[indx]
                if func_preds[indx] == 1:
                    for s in range(num_statements[indx]):
                        p = sample[s]
                        y_preds.append(p)
                else:
                    for _ in range(num_statements[indx]):
                        y_preds.append(0)
            labels = labels.cpu().numpy().tolist()
            for indx in range(len(labels)):
                sample = labels[indx]
                for s in range(num_statements[indx]):
                    lab = sample[s]
                    y_trues.append(lab)
            
    model.train()
    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    logger.info("***** Eval results *****")
    logger.info(f"F1 Accuracy: {str(f1)}")
    return f1

def test(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []
    func_level_trues = []
    func_level_preds = []
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (input_ids, statement_mask, labels, func_labels, num_statements) = [x.to(args.device) for x in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                        statement_mask=statement_mask)
            preds = torch.where(probs>0.5, 1, 0).tolist()
            
            func_preds = torch.argmax(func_probs, dim=-1).tolist()
            
            ### function-level ###
            func_labels = func_labels.cpu().numpy().tolist()
            func_level_trues += func_labels
            func_level_preds += func_preds
            
            for indx in range(len(preds)):
                sample = preds[indx]
                if func_preds[indx] == 1:
                    for s in range(num_statements[indx]):
                        p = sample[s]
                        y_preds.append(p)
                else:
                    for _ in range(num_statements[indx]):
                        y_preds.append(0)
            labels = labels.cpu().numpy().tolist()
            for indx in range(len(labels)):
                sample = labels[indx]
                for s in range(num_statements[indx]):
                    lab = sample[s]
                    y_trues.append(lab)
            
    f1 = f1_score(y_true=func_level_trues, y_pred=func_level_preds)
    acc = accuracy_score(y_true=func_level_trues, y_pred=func_level_preds)
    recall = recall_score(y_true=func_level_trues, y_pred=func_level_preds)
    pre = precision_score(y_true=func_level_trues, y_pred=func_level_preds)

    logger.info("***** Function-level Test results *****")
    logger.info(f"F1 Score: {str(f1)}")
    logger.info(f"acc Score: {str(acc)}")
    logger.info(f"recall Score: {str(recall)}")
    logger.info(f"pre Score: {str(pre)}")

    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    recall = recall_score(y_true=y_trues, y_pred=y_preds)
    pre = precision_score(y_true=y_trues, y_pred=y_preds)

    logger.info("***** Line-level Test results *****")
    logger.info(f"F1 Score: {str(f1)}")
    logger.info(f"acc Score: {str(acc)}")
    logger.info(f"recall Score: {str(recall)}")
    logger.info(f"pre Score: {str(pre)}")
    return f1

def calculate_line_metrics(pred, target):
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples')
            }

def calculate_func_metrics(pred, target):
    return {'accuracy': accuracy_score(y_true=target, y_pred=pred),
            'precision': precision_score(y_true=target, y_pred=pred),
            'recall': recall_score(y_true=target, y_pred=pred),
            'f1': f1_score(y_true=target, y_pred=pred),}

def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--eval_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--test_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--pretrain_language", default="", type=str, required=False,
                        help="python, go, ruby, php, javascript, java, c_cpp")
    ps.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ps.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    ps.add_argument("--encoder_block_size", default=512, type=int,
                        help="")
    ps.add_argument("--max_line_length", default=64, type=int,
                        help="")
    ps.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    ps.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    ps.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    ps.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    ps.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path") 
    ps.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    ps.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    ps.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    ps.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    ps.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    ps.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    ps.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for AdamW.")
    ps.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    ps.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    ps.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    ps.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    ps.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    ps.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    ps.add_argument('--epochs', type=int, default=3,
                        help="training epochs")
    args = ps.parse_args()

    args.num_labels = 155

    # Setup CUDA, GPU
    args.n_gpu = 1
    args.device = "cuda:0"
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    # Set seed
    set_seed(args)
    
    tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base") # "Salesforce/codet5p-220m"
    xfmr = T5EncoderModel.from_pretrained("Salesforce/codet5-base") # "Salesforce/codet5p-220m"
    
    model = Model(xfmr, tokenizer, args, hidden_dim=768)
    logger.info("Training/evaluation parameters %s", args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    # Training
    if args.do_train:
        output_dir = "./saved_models/checkpoint-best-f1/codet5_base_big_vul.bin"
        model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)   
        
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='val')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset)


if __name__ == "__main__":
    main()
