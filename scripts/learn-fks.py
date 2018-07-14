#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random, time, sys, signal, copy, math
import numpy as np

import chainer
from chainer import Variable, Link, Chain, ChainList, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.mean_squared_error import mean_squared_error

from matplotlib import pyplot as plt
import argparse

import rospy
from fk_with_nn.srv import *
from std_msgs.msg import Float64MultiArray

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(5, 20),
            l2=L.Linear(50),
            l3=L.Linear(18)
            )
        
    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        o = self.l3(h)
        return o

    # forward and save output
    def __call__(self, x):
        self.res = self.forward(x)

    # loss func for learning
    def loss(self, t):
        return F.mean_squared_error(self.res, t)

    # loss for optimize input
    def loss_for_end_effector(self, t):
        #return F.mean_squared_error(self.res, t)        
        return Variable(np.array(((self.res - t)[0][-1].data**2 + (self.res - t)[0][-2].data**2 + (self.res - t)[0][-3].data**2) / 3).astype(np.float32))

class LearnFks():
    def __init__(self):
        self.TRAIN_TEST_RATIO = 1.0
        self.BATCH_SIZE = 1000

        self.X = []
        self.Y = []

        # init ros variable
        rospy.init_node("fk_server")
        rospy.Subscriber("angle_vector", Float64MultiArray, self.callback)
        r = rospy.Rate(30)        

        self.angle_vector = [0, 0, 0, 0, 0]

    def callback(self, msg):
        self.angle_vector = msg.data

    # print percentage of progress every 10%
    def percentage(self, idx, loop_num, val=None):
        split_num = 10
        if (idx + 1) % int(loop_num / split_num) == 0:
            if val == None:
                print('{}% {}/{}'.format((idx + 1) * 100 / loop_num, idx + 1, loop_num))
            else:
                print('{}% {}/{}, {}'.format((idx + 1) * 100 / loop_num, idx + 1, loop_num, val))

    # load data from log files
    def load_data(self, log_file="../log/fks.log"):
        with open(log_file, "r") as f:
            for l in f.readlines():
                val = l.split(' ')
                
                x = []
                x.append(float(val[0]))
                x.append(float(val[1]))
                x.append(float(val[2]))
                x.append(float(val[3]))
                x.append(float(val[4]))
                self.X.append(x)
                
                y = []
                y.append(float(val[5]))
                y.append(float(val[6]))
                y.append(float(val[7]))
                y.append(float(val[8]))
                y.append(float(val[9]))
                y.append(float(val[10]))
                y.append(float(val[11]))
                y.append(float(val[12]))
                y.append(float(val[13]))
                y.append(float(val[14]))
                y.append(float(val[15]))
                y.append(float(val[16]))
                y.append(float(val[17]))
                y.append(float(val[18]))
                y.append(float(val[19]))
                y.append(float(val[20]))
                y.append(float(val[21]))
                y.append(float(val[22]))                
                self.Y.append(y)
                
    # make NeuralNetwork model by using MyChain class
    def make_model(self):
        self.model = MyChain()
        self.optimizer = optimizers.RMSprop(lr=0.01)
        self.optimizer.setup(self.model)

    def get_batch_train(self, n):
        x = []
        y = []
        for i in range(n):
            r = (int(random.random() * 10 * len(self.X) * self.TRAIN_TEST_RATIO) % int(len(self.X) * self.TRAIN_TEST_RATIO))
            x.append(self.X[r])
            y.append(self.Y[r])
        return np.array(x), np.array(y)
    
    def get_test(self, log_file="../log/fks-test.log"):
        X = []
        Y = []
        with open(log_file, "r") as f:
            for l in f.readlines():
                val = l.split(' ')
                
                x = []
                x.append(float(val[0]))
                x.append(float(val[1]))
                x.append(float(val[2]))
                x.append(float(val[3]))
                x.append(float(val[4]))
                X.append(x)
                
                y = []
                y.append(float(val[5]))
                y.append(float(val[6]))
                y.append(float(val[7]))
                y.append(float(val[8]))
                y.append(float(val[9]))
                y.append(float(val[10]))
                y.append(float(val[11]))
                y.append(float(val[12]))
                y.append(float(val[13]))
                y.append(float(val[14]))
                y.append(float(val[15]))
                y.append(float(val[16]))
                y.append(float(val[17]))
                y.append(float(val[18]))
                y.append(float(val[19]))
                y.append(float(val[20]))
                y.append(float(val[21]))
                y.append(float(val[22]))                
                Y.append(y)
        
        return np.array(X), np.array(Y)

    # load trained NeuralNetwork model
    def load_model(self, log_file='../models/mymodel.h5'):
        serializers.load_hdf5(log_file, self.model)
        
    # save trained NeuralNetwork model
    def save_model(self):
        serializers.save_hdf5('../models/mymodel.h5', self.model)
        
    def train(self, loop_num=1000):
        losses = []
        now_loss = None
        for i in range(loop_num):
            self.percentage(i, loop_num, now_loss)

            # train
            x, y = self.get_batch_train(self.BATCH_SIZE)
        
            x_ = Variable(x.astype(np.float32).reshape(self.BATCH_SIZE, 5))
            t_ = Variable(y.astype(np.float32).reshape(self.BATCH_SIZE, 18))

            self.model.zerograds()
            self.model(x_)

            loss = self.model.loss(t_)
            now_loss = loss.data
                        
            loss.backward()
            self.optimizer.update()

            losses.append(loss.data)
            
        print('[Train] loss is {}'.format(losses[-1]))

        plt.title('Loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')                
        plt.plot(losses)
        plt.yscale('log')
        plt.show()
            
    def test(self):
        x, y = self.get_test()
        x_ = Variable(x.astype(np.float32).reshape(len(x),5))
        t_ = Variable(y.astype(np.float32).reshape(len(y),18))
        
        self.model.zerograds()
        self.model(x_)
        loss = self.model.loss(t_)
        
        print('[Test] loss is {}'.format(loss.data))
        

    def begin_fk(self):
        s = rospy.Service('fk', Fk, self.calcFK)
        print "Ready to FK."
        rospy.spin()
        
    def calcFK(self, req):
        x_ = Variable(np.array([[req.joint_angle[0], req.joint_angle[1], req.joint_angle[2], req.joint_angle[3], req.joint_angle[4]]]).astype(np.float32).reshape(1, 5))
        li.model(x_)
        res = li.model.res

        fk_res = FkResponse()
        fk_res.pos_x = [res[0][0].data, res[0][3].data, res[0][6].data, res[0][9].data, res[0][12].data, res[0][15].data]
        fk_res.pos_y = [res[0][1].data, res[0][4].data, res[0][7].data, res[0][10].data, res[0][13].data, res[0][16].data]
        fk_res.pos_z = [res[0][2].data, res[0][5].data, res[0][8].data, res[0][11].data, res[0][14].data, res[0][17].data]
        return fk_res

    def begin_move(self):
        s = rospy.Service('ik', Ik, self.optimize_for_end_effector)
        print "Ready to FK."
        rospy.spin()

    def optimize_for_end_effector(self, req):
        x = Variable(np.array([self.angle_vector]).astype(np.float32).reshape(1, 5))
        t = Variable(np.array([[req.pos_x[0], req.pos_y[0], req.pos_z[0],
                               req.pos_x[1], req.pos_y[1], req.pos_z[1],
                               req.pos_x[2], req.pos_y[2], req.pos_z[2],
                               req.pos_x[3], req.pos_y[3], req.pos_z[3],
                               req.pos_x[4], req.pos_y[4], req.pos_z[4],
                               req.pos_x[5], req.pos_y[5], req.pos_z[5]]]
        ).astype(np.float32).reshape(1, 18))
        
        # optimize loop
        for i in range(100):
            self.model.zerograds()
            self.model(x)
            t = Variable(np.array([[
                self.model.res[0][0].data, self.model.res[0][1].data, self.model.res[0][2].data,
                self.model.res[0][3].data, self.model.res[0][4].data, self.model.res[0][5].data,
                self.model.res[0][6].data, self.model.res[0][7].data, self.model.res[0][8].data,
                self.model.res[0][9].data, self.model.res[0][10].data, self.model.res[0][11].data,
                self.model.res[0][12].data, self.model.res[0][13].data, self.model.res[0][14].data,
                req.pos_x[5], req.pos_y[5], req.pos_z[5]]]).astype(np.float32).reshape(1, 18))
            loss = self.model.loss(t)
            loss.backward()
            x = Variable((x - 0.05 * x.grad_var).data)
            
            # apply input restriction
            for j in range(5):
                if x[0][j].data > 120:
                    x[0][j].data = np.float32(120)
                if x[0][j].data < -120:
                    x[0][j].data = np.float32(-120)

        ik_res = IkResponse()
        ik_res.joint_angle = [x[0][0].data, x[0][1].data, x[0][2].data, x[0][3].data, x[0][4].data]
        return ik_res
                    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))

    # init arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", nargs='?', default=False, const=True, help="train NN")
    parser.add_argument("--fk", "-f", nargs='?', default=False, const=True, help="forward kinematics")
    parser.add_argument("--move", "-move", nargs='?', default=False, const=True, help="move with bp")
    parser.add_argument("--model", "-m", nargs='?', default=False, const=True, help="model of network")        
    args = parser.parse_args()
    
    # parse
    train_flag = int(args.train)
    fk_flag = int(args.fk)
    model_flag = int(args.model)
    move_flag = int(args.move)    

    li = LearnFks()
    li.make_model()
    if model_flag:
        li.load_model()

    if train_flag:
        li.load_data()    
        print('[Train] start')
        li.train(loop_num=10000)
        li.save_model()

        print('[Test] start')            
        li.test()

    if fk_flag:        
        li.begin_fk()

    if move_flag:
        li.begin_move()
