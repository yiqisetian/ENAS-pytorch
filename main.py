"""Entry point."""
import os

import torch

import data
import config
import utils
import trainer

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)#data_dir="./data/ptb"

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        #Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'rnn':
        dataset = data.text.Corpus(args.data_path)#将文本数据读入字典，生成词对应的序号的Tensor
    elif args.dataset == 'cifar':
        dataset = data.image.Image(args.data_path)
    else:
        raise NotImplementedError("{}is not supported".format(args.dataset))

    trnr = trainer.Trainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in `derive` mode")
        trnr.derive()
    elif args.mode == 'test':
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trnr.test()
    elif args.mode == 'single':
        if not args.dag_path:
            raise Exception("[!] You should specify `dag_path` to load a dag")
        utils.save_args(args)
        trnr.train(single=True)
    else:
        raise Exception("[!] Mode not found: {}".format(args.mode))

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
