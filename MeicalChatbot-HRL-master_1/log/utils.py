# -*- coding : utf8 -*-
import os


def get_dir_list(path, key_word_list=None, no_key_word_list=None):
    file_name_list = os.listdir(path)  # 获得原始json文件所在目录里面的所有文件名称
    if key_word_list == None and no_key_word_list == None:
        temp_file_list = file_name_list
    elif key_word_list != None and no_key_word_list == None:
        temp_file_list = []
        for file_name in file_name_list:
            have_key_words = True
            for key_word in key_word_list:
                if key_word not in file_name:
                    have_key_words = False
                    break
                else:
                    pass
            if have_key_words == True:
                temp_file_list.append(file_name)
    elif key_word_list == None and no_key_word_list != None:
        temp_file_list = []
        for file_name in file_name_list:
            have_no_key_word = False
            for no_key_word in no_key_word_list:
                if no_key_word in file_name:
                    have_no_key_word = True
                    break
            if have_no_key_word == False:
                temp_file_list.append(file_name)
    elif key_word_list != None and no_key_word_list != None:
        temp_file_list = []
        for file_name in file_name_list:
            have_key_words = True
            for key_word in key_word_list:
                if key_word not in file_name:
                    have_key_words = False
                    break
                else:
                    pass
            have_no_key_word = False
            for no_key_word in no_key_word_list:
                if no_key_word in file_name:
                    have_no_key_word = True
                    break
                else:
                    pass
            if have_key_words == True and have_no_key_word == False:
                temp_file_list.append(file_name)
    print(key_word_list, len(temp_file_list))
    # time.sleep(2)
    return temp_file_list