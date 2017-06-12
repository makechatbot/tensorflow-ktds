# -*- coding: utf-8 -*-

import tensorflow as tf
import random
import math
import os

from config import FLAGS
from model import Seq2Seq
from dialog import Dialog


def train(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print ("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())


        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        total_batch = int(math.ceil(len(dialog.examples)/float(batch_size)))




        for step in range(total_batch * epoch):
            # option
            enc_input, dec_input, targets = dialog.next_batch(batch_size)

            # train
            _, loss = model.train(sess, enc_input, dec_input, targets)


        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print ('최적화 완료!')


def test(dialog, batch_size=100):
    print ("\n=== 예측 테스트 ===")

    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print ("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        enc_input, dec_input, targets = dialog.next_batch(batch_size)

        expect, outputs, accuracy = model.test(sess, enc_input, dec_input, targets)

        expect = dialog.decode(expect)
        outputs = dialog.decode(outputs)

        pick = random.randrange(0, len(expect) / 2)
        input = dialog.decode([dialog.examples[pick * 2]], True)
        expect = dialog.decode([dialog.examples[pick * 2 + 1]], True)
        outputs = dialog.cut_eos(outputs[pick])

        print("\n정확도:", accuracy)
        print("랜덤 결과\n",)
        print("    입력값:", input)
        print("    실제값:", expect)
        print("    예측값:", ' '.join(outputs))


def main(_):
    dialog = Dialog()

    dialog.load_vocab(FLAGS.voc_path)
    dialog.load_examples(FLAGS.data_path)

    if FLAGS.train:
        train(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
    elif FLAGS.test:
        test(dialog, batch_size=FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()
