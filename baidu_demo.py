import string
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate, baidu_raw_dataset
from model import Model
from src.baidudataset import BaiduCollate


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

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
    # AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    # demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    # demo_loader = torch.utils.data.DataLoader(
    #     demo_data, batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=int(opt.workers),
    #     collate_fn=AlignCollate_demo, pin_memory=True)

    demo_data = baidu_raw_dataset(root=opt.image_folder, opt=opt)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=True,  
        num_workers=int(opt.workers),
        collate_fn=BaiduCollate(opt.imgH, keep_ratio=True), pin_memory=True)

    outf = open(opt.out_csv, 'w')
    # predict
    model.eval()
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

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        print('-' * 80)
        print('image_path\tpredicted_labels')
        print('-' * 80)
        for img_name, pred in zip(image_path_list, preds_str):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

            print(f'{img_name}\t{pred}')

            content = os.path.basename(img_name) + '\t' + pred +'\n'
            outf.write(content)
            outf.flush()
    outf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='/root/shenlan/deepblue/1_OCR/data/test_images/', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--saved_model', default='./saved_models/None-ResNet-BiLSTM-CTC-Seed1111/best_norm_ED.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='None', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    ''' result csv file '''
    parser.add_argument('--out_csv', type=str, default='./dataset/BAIDU/result.csv')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    with open('./dataset/BAIDU/baidu_alphabet.txt') as f:
            opt.character = f.readlines()[0]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
