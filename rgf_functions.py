# -*- coding: utf-8 -*-


__author__ = 'fucus'
import subprocess
import os
import numpy as np
import pandas as pd
import shutil
# RGF path
rgf_path = 'lib/rgf1.2'

def myDeleteDir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

def myMakeDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def start_RGF_train_test(xtrain,ytrain,wtrain,xtest,ytest,modelname,temp_dir,save_dir,param):

    # handle data input
    xtrain = np.array(xtrain)
    xtest = np.array(xtest)
    ytrain = np.array(ytrain)

    # write data
    data_dir = temp_dir + '/' + modelname +'_data'
    myMakeDir(data_dir)
    np.savetxt(data_dir + '/trainx.txt', xtrain, delimiter=' ')
    np.savetxt(data_dir + '/trainy.txt', ytrain, delimiter=' ')
    np.savetxt(data_dir + '/testx.txt', xtest, delimiter=' ')
    np.savetxt(data_dir + '/testy.txt', ytest, delimiter=' ')

    # write settings file
    output_dir = save_dir + '/' + modelname + '_output'
    with open(data_dir+'/'+modelname+'.inp', 'w') as fp:
        fp.write('train_x_fn=' + data_dir + '/trainx.txt\n')
        fp.write('train_y_fn=' + data_dir + '/trainy.txt\n')
        fp.write('test_x_fn=' + data_dir + '/testx.txt\n')
        fp.write('test_y_fn=' + data_dir + '/testy.txt\n')
        fp.write('evaluation_fn=' + data_dir + '/output.evaluation\n')
        fp.write('model_fn_prefix=' + output_dir + '/m\n')
        for p in param.items():
            fp.write("%s=%s\n" % p)

        fp.write('SaveLastModelOnly\n')
        fp.write('Verbose')
        fp.close()

    # start RGF
    myMakeDir(output_dir)
    p = subprocess.Popen('perl ' + rgf_path + '/test/call_exe.pl ' + rgf_path + '/bin/rgf train_test ' + data_dir + '/' + modelname,
                         shell=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return p

def start_RGF_train(xtrain,ytrain,wtrain,modelname,temp_dir,save_dir,param):

    # handle data input
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    # write data
    data_dir = temp_dir + '/' + modelname +'_data'
    myMakeDir(data_dir)
    np.savetxt(data_dir + '/trainx.txt', xtrain, delimiter=' ')
    np.savetxt(data_dir + '/trainy.txt', ytrain, delimiter=' ')

    # write settings file
    output_dir = save_dir + '/' + modelname + '_output'
    with open (data_dir+'/'+modelname+'.inp', 'w') as fp:
        fp.write('train_x_fn=' + data_dir + '/trainx.txt\n')
        fp.write('train_y_fn=' + data_dir + '/trainy.txt\n')
        fp.write('model_fn_prefix=' + output_dir + '/m\n')

        for p in param.items():
            fp.write("%s=%s\n" % p)
        fp.write('SaveLastModelOnly\n')
        fp.write('Verbose')
        fp.close()

    # start RGF
    myMakeDir(output_dir)
    p = subprocess.Popen('perl ' + rgf_path + '/test/call_exe.pl ' + rgf_path + '/bin/rgf train ' + data_dir + '/' + modelname,
                         shell=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return p

def startTraining(train_dat,train_target,param, train_weights=None,modelname= "test_model",temp_dir="output/RGF_temp/",save_dir="output/RGF_save_temp/",predict_weights=False):

    myDeleteDir(temp_dir)
    myDeleteDir(save_dir)

    n = len(train_dat)
    # start full models
    print("starting RGF models and waiting for them to finish, this can take a while")
    procs = []
    procs.append(start_RGF_train(train_dat, train_target ,train_weights,modelname+'_full',temp_dir,save_dir,param))

    procs=[]
    # display output & wait to finish
    print("will print RGF output to console")
    for p in procs:
        while p.poll() is None:
            output = p.stdout.readline()
            print(output)


    print("rgf done")




def makePredictions(test_dat,temp_dir="output/RGF_temp/",save_dir="output/RGF_save_temp/"):
    # write data
    data_dir = temp_dir + '/test_data'
    myMakeDir(data_dir)
    np.savetxt(data_dir + '/testx.txt', test_dat, delimiter=' ')

    # make predictions for each model file in the model output folder
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.startswith("m-") and len(file)==4:
                with open (temp_dir+'/temp_pred.inp', 'w') as fp:
                    modelfile = os.path.join(root, file)
                    dirname = os.path.basename(os.path.normpath(root))
                    fp.write('test_x_fn=' + data_dir + '/testx.txt\n')
                    fp.write('model_fn=' + modelfile + '\n')
                    fp.write('prediction_fn=' + modelfile + '.pred')
                    fp.close()
                    p = subprocess.Popen('perl ' + rgf_path + '/test/call_exe.pl ' + rgf_path + '/bin/rgf predict ' + temp_dir + '/temp_pred',
                         shell=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    while p.poll() is None:
                        output = p.stdout.readline()
                        print(output)

if __name__ == '__main__':
    train_dat = [
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],

        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],


        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]
    train_target = [
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
    ]
    train_weights = None
    modelname = "test_model"
    temp_dir = "output/RGF_temp/"
    save_dir = "output/RGF_save_temp/"
    param = {"reg_L2": 1}
    predicts_weights = None

    test_dat = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]
    print(type(train_dat))
    startTraining(train_dat,train_target,train_weights,modelname,temp_dir,save_dir,param)
    makePredictions(test_dat,temp_dir,save_dir)

#class RegularizedGreedyForest(Model):
