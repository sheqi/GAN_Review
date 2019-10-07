import sys
import time
from argparse import ArgumentParser
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from action_recognition.model import create_model
from action_recognition.spatial_transforms import (CenterCrop, Compose,
                                                   Normalize, Scale, ToTensor, MEAN_STATISTICS, STD_STATISTICS)
from action_recognition.utils import load_state, generate_args
from action_recognition.options import BoolFlagAction
from action_recognition.temporal_transforms import (
    LoopPadding, TemporalRandomCrop, TemporalStride)

TEXT_COLOR = (255, 255, 255)
TEXT_FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_FONT_SIZE = 1
TEXT_VERTICAL_INTERVAL = 45
NUM_LABELS_TO_DISPLAY = 2

class TorchActionRecognition:
    def __init__(self, encoder, checkpoint_path, cuda, num_classes=400):
        model_type = "{}_vtn".format(encoder)

        args, _ = generate_args(model=model_type, n_classes=num_classes, layer_norm=False, cuda=cuda)
        #print(checkpoint_path)
        args.pretrain_path = checkpoint_path
        self.model, _ = create_model(args, model_type)

        if cuda:
        	self.model = self.model.module
        	self.model.eval()
        	self.model.cuda()
        	# we train on GPU thus it automatically loads to GPU
        	checkpoint = torch.load(str(checkpoint_path))
        else:
        	self.model.eval()
        	checkpoint = torch.load(str(checkpoint_path), map_location=lambda storage, loc: storage)
		
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.model.load_checkpoint(checkpoint['state_dict'])
        #print(self.model)
        #load_state(self.model, checkpoint['state_dict'])
        self.preprocessing = make_preprocessing(args)
        #elf.tp_preprocessing = Compose([args.temporal_stride, LoopPadding(args.sample_duration / args.temporal_stride)])

        self.embeds = deque(maxlen=(args.sample_duration * args.temporal_stride))

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.preprocessing(frame)

    def infer_frame(self, frame):
        embedding = self._infer_embed(frame)
        self.embeds.append(embedding)
        sequence = self.get_seq()
        return self._infer_logits(sequence)

    def _infer_embed(self, frame):
        with torch.no_grad():
            frame_tensor = frame.unsqueeze(0).to(self.device)
            tensor = self.model.resnet(frame_tensor)
            tensor = self.model.reduce_conv(tensor)
            embed = F.avg_pool2d(tensor, 7)
        return embed.squeeze(-1).squeeze(-1)

    def _infer_logits(self, embeddings):
        with torch.no_grad():
            ys = self.model.self_attention_decoder(embeddings)
            ys = self.model.fc(ys)
            ys = ys.mean(1)
        return ys.cpu()

    def _infer_seq(self, frame):
        with torch.no_grad():
            result = self.model(frame.view(1, 16, 3, 224, 224).to(self.device))
        return result.cpu()

    def get_seq(self):
        sequence = torch.stack(tuple(self.embeds), 1)
        #print(sequence.size())
        sequence = sequence[:, ::2, :]
        #print(sequence.size())
        if sequence.size(1) < 16:
            num_repeats = 15 // sequence.size(1) + 1
            sequence = sequence.repeat(1, num_repeats, 1)[:, :16, :]
        #print(sequence.size())
        return sequence

def make_preprocessing(args):
    return Compose([
        Scale(args.sample_size),
        CenterCrop(args.sample_size),
        ToTensor(args.norm_value),
        Normalize(MEAN_STATISTICS[args.mean_dataset], STD_STATISTICS[args.mean_dataset])
    ])


def draw_rect(image, bottom_left, top_right, color=(0, 0, 0), alpha=1.):
    xmin, ymin = bottom_left
    xmax, ymax = top_right

    image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :] * (1 - alpha) + np.asarray(color) * alpha
    return image


def render_frame(frame, probs, labels):
    order = probs.argsort(descending=True)

    status_bar_coorinates = (
        (0, 0),  # top left
        (650, 25 + TEXT_VERTICAL_INTERVAL * NUM_LABELS_TO_DISPLAY)  # bottom right
    )

    draw_rect(frame, status_bar_coorinates[0], status_bar_coorinates[1], alpha=0.5)

    for i, imax in enumerate(order[:NUM_LABELS_TO_DISPLAY]):
        text = '{} - {:.1f}%'.format(labels[imax], probs[imax] * 100)
        text = text.upper().replace("_", " ")
        cv2.putText(frame, text, (15, TEXT_VERTICAL_INTERVAL * (i + 1)), TEXT_FONT_SIZE,
                    TEXT_FONT_FACE, TEXT_COLOR)

    return frame

def run_demo(model, video_cap, labels):
    tick = time.time()
    while video_cap.isOpened():
        ok, frame = video_cap.read()

        if not ok:
            break

        logits = model.infer_frame(model.preprocess_frame(frame))
        probs = F.softmax(logits[0])
        frame = render_frame(frame, probs, labels)

        tock = time.time()
        expected_time = tick + 1 / 30.
        if tock < expected_time:
            time.sleep(expected_time - tock)
        tick = tock

        cv2.imshow("demo", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

def run_demo_print(model, video_cap, labels):

    tick = time.time()
    all_frames = []
    while video_cap.isOpened():
        ok, frame = video_cap.read()
        if not ok:
            break
        logits = model.infer_frame(model.preprocess_frame(frame))
        probs = F.softmax(logits[0])

        tock = time.time()
        expected_time = tick + 1 / 30.
        if tock < expected_time:
            time.sleep(expected_time - tock)
        tick = tock

        order = probs.argsort(descending=True)

        text = '{} - {:.1f}%'.format(labels[order[0]], probs[order[0]] * 100)
        print(text)

def run_demo_img(model, video_cap, labels):

    tick = time.time()
    tick_0 = time.time()
    all_frames = []
    while video_cap.isOpened():
        ok, frame = video_cap.read()

        if not ok:
            break
        all_frames.append(frame)

        logits = model.infer_frame(model.preprocess_frame(frame))
        probs = F.softmax(logits[0])

        # default 30 fps data stream 
        # the inference wait for the upcoming frame
        tock_0 = time.time()
        expected_time = tick_0 + 1 /30.
        if tock_0 < expected_time:
        	time.sleep(expected_time - tock_0)
        tick_0 = tock_0 

        order = probs.argsort(descending=True)

        text = '{} - {:.1f}%'.format(labels[order[0]], probs[order[0]] * 100)
        print(text)

    tock = time.time()
    td = tock - tick
    print('The inference time is {:.0f}m {:.0f}s'.format(td // 60, td % 60))
    print('fps is {:.0f}'.format(len(all_frames) / td))

def main():
    parser = ArgumentParser(description="inference on the input streams.")

    parser.add_argument("--encoder", default='resnet34', help="What encoder to use ")
    parser.add_argument("--checkpoint", help="Path to pretrained model (.pth) file", required=True)
    parser.add_argument("--input-video", type=str, help="Path to input video", required=True)
    parser.add_argument("--labels", help="Path to labels file (new-line separated file with label names)", type=str, required=True)
    # use --no-cuda to disable
    parser.add_argument("--cuda", action=BoolFlagAction, default=True, help="Whether cuda should be used")
    parser.add_argument("-j", "--n-threads", default=4, type=int, help="Number of threads for multi-thread loading")

    args = parser.parse_args()

    with open(args.labels) as fd:
        labels = fd.read().strip().split('\n')
    model = TorchActionRecognition(args.encoder, args.checkpoint, args.cuda, num_classes=len(labels))
    
    cap = cv2.VideoCapture(args.input_video)
    
    # run_demo: with rendering for each image
    # run_demo(model, cap, labels)

    # run_demo_print: without rendering
    run_demo_print(model, cap, labels)

    # run_demo_img: demo with image sequence
    #run_demo_img(model, cap, labels)

if __name__ == '__main__':
    sys.exit(main())
