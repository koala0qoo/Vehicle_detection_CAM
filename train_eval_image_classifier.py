from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='')

    # train
    parser.add_argument('--dataset_name', type=str, default='quiz')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='inception_v4')
    parser.add_argument('--checkpoint_exclude_scopes', type=str, default='InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits')
    parser.add_argument('--train_dir', type=str, default='train')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clone_on_cpu', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--batch_size', type=int, default=32)

    # eval
    parser.add_argument('--dataset_split_name', type=str, default='validation')
    parser.add_argument('--eval_dir', type=str, default='validation')
    parser.add_argument('--max_num_batches', type=int, default=128)

    # inference
    parser.add_argument('--inference_size', type=int, default=1)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


train_cmd = 'python ./train_image_classifier.py  --dataset_name={dataset_name} --dataset_dir={dataset_dir} --model_name={model_name} --checkpoint_exclude_scopes={checkpoint_exclude_scopes} --train_dir={train_dir} --learning_rate={learning_rate} --optimizer={optimizer} --batch_size={batch_size} --max_number_of_steps={max_number_of_steps} --clone_on_cpu={clone_on_cpu}'
eval_cmd = 'python ./eval_image_classifier.py --dataset_name={dataset_name} --dataset_dir={dataset_dir} --dataset_split_name={dataset_split_name} --model_name={model_name}   --checkpoint_path={checkpoint_path}  --eval_dir={eval_dir} --batch_size={batch_size}  --max_num_batches={max_num_batches}'
export_cmd = 'python ./export_inference_graph.py --model_name={model_name} --dataset_name={dataset_name} --output_file={output_file} --dataset_dir={dataset_dir}'
freeze_graph_cmd = 'python ./freeze_graph.py --input_graph={input_graph} --input_checkpoint={input_checkpoint} --output_graph={output_graph} --input_binary={input_binary} --output_node_name={output_node_name}'
inference_cmd = 'python ./inference.py --output_dir={output_dir} --dataset_dir={dataset_dir} --inference_size={inference_size}'

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    step_per_epoch = 43971 // FLAGS.batch_size

    if FLAGS.checkpoint_path:
        ckpt = ' --checkpoint_path=' + FLAGS.checkpoint_path
    else:
        ckpt = ''
    for i in range(30):
        steps = int(step_per_epoch * (i + 1))
        # train 1 epoch
        print('################    train    ################')
        p = os.popen(train_cmd.format(**{'dataset_name': FLAGS.dataset_name, 'dataset_dir': FLAGS.dataset_dir,
                                         'model_name': FLAGS. model_name,
                                         'checkpoint_exclude_scopes': FLAGS.checkpoint_exclude_scopes, 'train_dir': FLAGS. train_dir,
                                         'learning_rate': FLAGS.learning_rate, 'optimizer': FLAGS.optimizer,
                                         'batch_size': FLAGS.batch_size, 'max_number_of_steps': steps, 'clone_on_cpu': FLAGS.clone_on_cpu}) + ckpt)
        for l in p:
            print(l.strip())

        # eval
        print('################    eval    ################')
        p = os.popen(eval_cmd.format(**{'dataset_name': FLAGS.dataset_name, 'dataset_dir': FLAGS.dataset_dir,
                                        'dataset_split_name': 'validation', 'model_name': FLAGS. model_name,
                                        'checkpoint_path': FLAGS.train_dir, 'batch_size': FLAGS.batch_size,
                                        'eval_dir': FLAGS. eval_dir, 'max_num_batches': FLAGS. max_num_batches}))
        for l in p:
            print(l.strip())
            
    '''
    print('################    export_inference_graph    ################')
    file_path = os.path.join(FLAGS.output_dir, 'exported_graphs/inference_graph.pb')
    p = os.popen(export_cmd.format(**{'model_name': FLAGS.model_name, 'dataset_name': FLAGS.dataset_name,
                                      'output_file': file_path, 'dataset_dir': FLAGS.dataset_dir}))
    file_path_ = os.path.join(FLAGS.output_dir, 'exported_graphs/frozen_inference_graph.pb')
    ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
    p = os.popen(freeze_graph_cmd.format(**{'input_graph': file_path, 'input_checkpoint': ckpt,
                                            'output_graph': file_path_, 'input_binary': True,
                                            'output_node_name': 'cam_classifier/A/conv3_1x1/Conv2D,cam_classifier/A/Flatten/flatten/Reshape'}))

    print('################    inference    ################')
    p = os.popen(inference_cmd.format(**{'output_dir': FLAGS.output_dir, 'dataset_dir': FLAGS.dataset_dir,
                                         'inference_size': FLAGS.inference_size}))
    '''
