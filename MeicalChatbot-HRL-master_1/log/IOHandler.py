# -*- coding: utf8 -*-
"""
写一个类，用来进行文件的读写操作。
"""

import datetime
import csv
import os


class FileIO(object):
    @staticmethod
    def writeToFile(text, filename):
        file = open(filename, 'a+',encoding='utf8')
        file.write(text + '\n')
        file.close()

    @staticmethod
    def writeToCsvFile(list_msg, filename, mode='a+'):
        file = open('./' + filename, mode=mode,encoding='utf8')
        writer = csv.writer(file)
        writer.writerow(list_msg)
        file.close()

    @staticmethod
    def exceptionHandler(message, url=''):
        FileIO.writeToFile(text='[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']: ' + url + '\n'
                             + message, filename='./../../logs/error_log.logs')