import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter, TransformerConverter
from dataset import RawDataset, AlignCollate
from model import Model

import Levenshtein
import shutil


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    elif 'Bert' in opt.Prediction:
        converter = TransformerConverter(opt.character, opt.batch_max_length)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    opt.alphabet_size = len(opt.character) + 2  # +2 for [UNK]+[EOS]

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    if opt.mode == "valid":
        pred_txt = open("/workspace/xwh/aster/train/val/train_paixu.txt", "r")
        lines = pred_txt.readlines()
        len_txt = len(lines)

        txt_1 = open("/workspace/xwh/github/2dattention/Bert_OCR.pytorch-master/wrong/txt_0_0.5.txt", "w")
        txt_2 = open("/workspace/xwh/github/2dattention/Bert_OCR.pytorch-master/wrong/txt_0.5_1.txt", "w")

    if opt.mode == "test":
        test_txt = open("/workspace/xwh/github/2dattention/Bert_OCR.pytorch-master/test/test_txt.txt", "w")

    # predict
    model.eval()
    count = 0
    correct = 0
    bb_count = 0
    for image_tensors, image_path_list in demo_loader:
        batch_size = image_tensors.size(0)
        with torch.no_grad():
            image = image_tensors.cuda()
            # For max length prediction
            length_for_pred = torch.cuda.IntTensor([opt.batch_max_length] * batch_size)
            text_for_pred = torch.cuda.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        elif 'Bert' in opt.Prediction:
            with torch.no_grad():
                pad_mask = None
                preds = model(image, pad_mask)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds[1].max(2)
                length_for_pred = torch.cuda.IntTensor([preds_index.size(-1)] * batch_size)
                preds_str = converter.decode(preds_index, length_for_pred)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        # print('-' * 80)
        # print('image_path\tpredicted_labels')
        # print('-' * 80)
        for img_name, pred in zip(image_path_list, preds_str):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

            if opt.mode == "valid":
                gt = lines[count].strip('\n').split(" ")[1]
                bb = Levenshtein.ratio(pred, gt)
                print(pred, gt, bb)
                if pred == gt:
                    correct += 1
                bb_count += bb
                print('Acc: {:.2%}'.format(correct / len_txt))
                print('edit_distance: {:.2%}'.format(bb_count / len_txt))
                count += 1

                if bb <= 0.5:
                    shutil.copy(img_name, "/workspace/xwh/github/2dattention/Bert_OCR.pytorch-master/wrong/image_0_0.5/")
                    newline = str(img_name) + " " + str(gt) + " " + str(pred) + " " + str(bb) + '\n'
                    txt_1.writelines(newline)

            if opt.mode == "test":
                count += 1
                num = '%06d' % count
                newline = "test_" + str(num) + ".jpg" + "," + str(pred) + "\n"
                test_txt.writelines(newline)
            print(f'{img_name}\t{pred}')


if __name__ == '__main__':
    character_list = open("./dataset/alphabet.txt", 'r')
    character_list = character_list.readline()
    # print(character_list)
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='demo_image/', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--saved_model', default='./saved_models/TPS-AsterRes-Bert-Bert_pred-Seed666/best_accuracy.pth', help="path to saved_model to evaluation")
    parser.add_argument('--mode', type=str, default='valid', help='mode. valid|test')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=40, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default=character_list, help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='AsterRes', help='FeatureExtraction stage. VGG|RCNN|ResNet|AsterRes')
    parser.add_argument('--SequenceModeling', type=str, default='Bert', help='SequenceModeling stage. None|BiLSTM|Bert')
    parser.add_argument('--Prediction', type=str, default='Bert_pred', help='Prediction stage. CTC|Attn|Bert_pred')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=1024,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--position_dim', type=int, default=210, help='the length sequence out from cnn encoder')
    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
