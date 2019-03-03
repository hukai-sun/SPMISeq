import logging
import sys
import uuid
import time
import datetime
import os
from os.path import join, exists
from os import makedirs

import torch 
import json

def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def path_exists(path):
    if not exists(path):
        makedirs(path)
    return path


def get_output_path():

    full_path = 'result'

    return path_exists(full_path)

def create_root_output_path(type,time_point):
    t = time_point
    d = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    root = 'id_%s_%s' % (str(d), type)
    root = join(type,root)
   
   # root = join(type,'id_%s' % str(d))

    path = join(get_output_path(), root)
    if exists(path): path += "_(%s)" % str(uuid.uuid4())
    return path_exists(path)
    
def get_logging_path(root_path,time_point):
    t = time_point
    n = "_logging_%s.log" % datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d-%H%M%S')
    return join(root_path, n)


def init_logging(model_name,time_point):
    model_name = model_name
    logger = logging.getLogger('%slogger' % model_name)
    for hdlr in logger.handlers: logger.removeHandler(hdlr)
    rootpaths = create_root_output_path(model_name,time_point)
    hdlr = logging.FileHandler(get_logging_path(rootpaths,time_point))
    formatter = logging.Formatter('%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger, rootpaths


def save_checkpoint(state, filename):
    """
    save checkpoint
    """
    torch.save(state, filename+'.model')


