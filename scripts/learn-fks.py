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

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(5, 20),
            l2=L.Linear(50),
            l3=L.Linear(50),
            l4=L.Linear(50),
            l5=L.Linear(18)
            )
        
    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        o = self.l5(h)
        return o

    # forward and save output
    def __call__(self, x):
        self.res = self.forward(x)

    # loss func for learning
    def loss(self, t):
        return F.mean_squared_error(self.res, t)

    # loss for optimize input
    def loss_for_optimize_input(self, t):
        return F.mean_squared_error(self.res, t)

class LearnFks():
    def __init__(self):
        self.TRAIN_TEST_RATIO = 1.0
        self.BATCH_SIZE = 1000

        self.X = []
        self.Y = []

        # init ros variable
        rospy.init_node("fk_server")

    # print percentage of progress every 10%
    def percentage(self, idx, loop_num):
        split_num = 10
        if (idx + 1) % int(loop_num / split_num) == 0:
            print('{}% {}/{}'.format((idx + 1) * 100 / loop_num, idx + 1, loop_num))

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
        for i in range(loop_num):
            self.percentage(i, loop_num)

            # train
            x, y = self.get_batch_train(self.BATCH_SIZE)
        
            x_ = Variable(x.astype(np.float32).reshape(self.BATCH_SIZE, 5))
            t_ = Variable(y.astype(np.float32).reshape(self.BATCH_SIZE, 18))

            self.model.zerograds()
            self.model(x_)

            loss = self.model.loss(t_)
                        
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
        
        print self.model.res
        print t_
        
        print('[Test] loss is {}'.format(loss.data))
        
    '''
    def optimize_input(self): # TODO add max input restriction
        x = Variable(np.array([self.past_states[-1 * self.DELTA_STEP], self.past_states[-2 * self.DELTA_STEP], self.now_input]).astype(np.float32).reshape(1,6)) # TODO random value is past state
        t = Variable(np.array(self.state_ref).astype(np.float32).reshape(1,2))
        loop_flag = True
        print "po"
        for i in range(20):    # optimize loop  loop_num is 10 == hz is 90
            self.model.zerograds()            
            self.model(x)
            # loss = self.model.loss(t)

            loss_pitch = 0
            for i in range(len(t)):
                loss_pitch += (t[i][0].data - self.model.res[i][0].data)**2
            loss_pitch /= len(t)
            loss_yaw = 0
            for i in range(len(t)):
                loss_yaw += (t[i][1].data - self.model.res[i][1].data)**2 #self.model.loss_yaw(t_test_)
            loss_yaw /= len(t)
 
            loss = self.model.loss_for_optimize_input(t)
            loss.backward()
            
            x = Variable((x - 0.0005 * x.grad_var).data)
            now_input = [x[0][4].data, x[0][5].data]
            print now_input[0], now_input[1], loss_pitch, loss_yaw
            # apply input restriction
            for j in range(self.PAST_INPUT_NUM * self.INPUT_DIM):
                # diff input restriction
                if now_input[j] - self.now_input[j] > self.MAX_INPUT_DIFF_RESTRICTION[j]:
                    now_input[j] = np.float32(self.now_input[j] + self.MAX_INPUT_DIFF_RESTRICTION[j])
                    #loop_flag = False
                elif self.now_input[j] - now_input[j] > self.MAX_INPUT_DIFF_RESTRICTION[j]:
                    now_input[j] = np.float32(self.now_input[j] - self.MAX_INPUT_DIFF_RESTRICTION[j])
                    #loop_flag = False                    
                # max min input restriction
                if now_input[j] > self.MAX_INPUT_RESTRICTION[j]:
                    now_input[j] = self.MAX_INPUT_RESTRICTION[j]
                    #loop_flag = False                    
                elif now_input[j] < self.MIN_INPUT_RESTRICTION[j]:
                    now_input[j] = self.MIN_INPUT_RESTRICTION[j]
                    #loop_flag = False                    
              
            x = Variable(np.array([self.past_states[-1 * self.DELTA_STEP], self.past_states[-2 * self.DELTA_STEP], now_input]).astype(np.float32).reshape(1,6)) # TODO random value is past state
            if loop_flag == False:
                break

        self.now_input = [float(x[0][self.PAST_STATE_NUM * self.STATE_DIM + 0].data), float(x[0][self.PAST_STATE_NUM * self.STATE_DIM + 1].data)]
        self.past_inputs.append([self.now_input[0], self.now_input[1]])
    '''

    def begin(self):
        s = rospy.Service('fk', Fk, self.calcFK)
        print "Ready to FK."
        rospy.spin()
        
    def calcFK(self, req):
        x_ = Variable(np.array([[req.a, req.b, req.c, req.d, req.e]]).astype(np.float32).reshape(1, 5))
        li.model(x_)
        res = li.model.res
   
        return FkResponse(
            res[0][0].data,
            res[0][1].data,
            res[0][2].data,
            res[0][3].data,
            res[0][4].data,
            res[0][5].data,
            res[0][6].data,
            res[0][7].data,            
            res[0][8].data,
            res[0][9].data,
            res[0][10].data,
            res[0][11].data,           
            res[0][12].data,
            res[0][13].data,
            res[0][14].data,
            res[0][15].data,
            res[0][16].data,
            res[0][17].data 
        )
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))

    # init arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", nargs='?', default=False, const=True, help="train NN")
    parser.add_argument("--fk", "-f", nargs='?', default=False, const=True, help="forward kinematics")
    parser.add_argument("--model", "-m", nargs='?', default=False, const=True, help="model of network")        
    args = parser.parse_args()
    
    # parse
    train_flag = int(args.train)
    fk_flag = int(args.fk)
    model_flag = int(args.model)

    li = LearnFks()
    li.make_model()
    if model_flag:
        li.load_model()
    li.load_data()    

    if train_flag:    
        print('[Train] start')
        li.train(loop_num=10000)
        li.save_model()

        print('[Test] start')            
        li.test()

    if fk_flag:        
        li.begin()
