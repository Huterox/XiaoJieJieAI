"""
for building corpus for chatboot running
This will be deployed in a white-hole, possibly in version 0.7
"""
import pickle
from tqdm import tqdm
from config import config
from utils.cut_word import Cut

class Chat_corpus(object):

    def __init__(self):
        self.Cut = Cut()
        self.PAD = 'PAD'
        self.UNKNOW = 'UNKNOW'
        self.EOS = 'EOS'
        self.SOS = 'SOS'
        self.word2index={
            self.PAD: config.chatboot_config.get("padding_idx"),
            self.SOS: config.chatboot_config.get("sos_idx"),
            self.EOS: config.chatboot_config.get("eos_idx"),
            self.UNKNOW: config.chatboot_config.get("unk_idx"),
        }
        self.index2word = {}
        self.count = {}

    def fit(self,sentence_list):
        """
        just for counting word
        :param sentence_list:
        :return:
        """
        for word in sentence_list:
            self.count[word] = self.count.get(word,0)+1

    def build_vocab_chat(self,min_count=None,max_count=None,max_feature=None):
        """
        build word dict,this need to save by pickle in computer memory
        :return:
        """

        temp = self.count.copy()
        for key in temp:
            cur_count = self.count.get(key,0)
            if(min_count !=None):
                if(cur_count<min_count):
                    del self.count[key]

            if(max_count!=None):
                if(cur_count>max_count):
                    del self.count[key]

        if(max_feature!=None):
            self.count = dict(sorted(self.count.items(),key= lambda x:x[1],
                                      reverse=True
                                      )[:max_feature]
                               )

        for key in self.count:
            self.word2index[key] = len(self.word2index)
        self.index2word = {item[1]:item[0] for item in self.word2index.items()}

    def transform(self,sentence,max_len,add_eos=False):
        if(len(sentence)>max_len):
            sentence = sentence[:max_len]
        sentence_len = len(sentence)
        if(add_eos):
            sentence = sentence+[self.EOS]
        if(sentence_len<max_len):
            sentence = sentence +[self.PAD]*(max_len-sentence_len)
        result = [self.word2index.get(i,self.word2index.get(self.UNKNOW)) for i in sentence]
        return result

    def inverse_transform(self,indices):
        """
        index ---> sentence
        :param indices:
        :return:
        """
        result = []
        for i in indices:
            if(i==self.word2index.get(self.EOS)):
                break
            result.append(self.index2word.get(i,self.UNKNOW))
        return result

    def __len__(self):
        return len(self.word2index)

    def __by_word(self,data_lines):
        for line in data_lines:
            for word in self.Cut.cut(line,by_word=True):
                self.word2index[word] = self.word2index.get(word,0)+1

    def __by_not_word(self,data_lines):
        for line in  data_lines:
            for word in self.Cut.cut(line,by_word=False):
                self.word2index[word] = self.word2index.get(word, 0) + 1

    def division(self,by_word=False,use_stop_word=False):
        """
        this funcation just for dividing input and target in xiaohuangji corpus
        :return:
        """
        count_input = 0
        count_target = 0
        temp_sentence = []

        if(by_word):
            middle_prx = ""
        else:
            middle_prx = "_no"

        target_save = open(config.chatboot_config.get("target_path"+middle_prx+"_by_word"),'a',encoding='utf-8')
        input_save  = open(config.chatboot_config.get("input_path"+middle_prx+"_by_word"),'a',encoding='utf-8')
        xiaohuangji_path = config.data_path.get("xiaohuangji")

        with open(xiaohuangji_path,'r',encoding='utf-8') as file:
            file_lines = tqdm(file.readlines(),desc="division xiaohuangji")
            for line in file_lines:
                line = line.strip()
                if (line.startswith("E")):
                    continue
                elif (line.startswith("M")):
                    line = line[1:].strip()
                    line = self.Cut.cut(line, by_word, use_stop_word)
                    temp_sentence.append(line)

                if(len(temp_sentence)==2):
                    """
                    Because the special symbol has a certain possibility, 
                    it is used as the input of the user.
                    Therefore, retain that special kind of "symbolic dialogue" corpus
                    """
                    if(len(line)==0):
                        temp_sentence = []
                        continue
                    input_save.write(" ".join(line)+'\n')
                    count_input+=1
                    target_save.write(" ".join(line)+'\n')
                    count_target+=1
                    temp_sentence=[]
            input_save.close()
            target_save.close()
            assert count_target==count_input,'count_target need equal count_input'
            print("\033[0;32;40m process is finished!\033[0m")
            print("The input len is:",count_input,"\nThe target len is:",count_target)



def compute_build(chat_corpus,fixed=False,
                  by_word=False,min_count=5,
                  max_count=None,max_feature=None,
                  is_target=True,
                  ):
    """
    for computing fit function with input and target file
    :param fixed: if True when error coming will try to fix by itself
    :return:
    """

    if (by_word):
        middle_prx = ""
    else:
        middle_prx = "_no"


    after_fixed = False
    lines = []

    try:
        if(is_target):
            lines = open(config.chatboot_config.get("target_path"+middle_prx+"_by_word"), 'r', encoding='utf-8').readlines()
        else:
            lines = open(config.chatboot_config.get("input_path"+middle_prx+"_by_word"), 'r', encoding='utf-8').readlines()
    except Exception as e:
        if(fixed):
            chat_corpus.division(by_word=by_word)
            after_fixed = True
        else:
            raise Exception("you need use Chat_corpus division function first! ")

    if(after_fixed):
        if (is_target):
            lines = open(config.chatboot_config.get("target_path" + middle_prx + "_by_word"), 'r',
                         encoding='utf-8').readlines()
        else:
            lines = open(config.chatboot_config.get("input_path" + middle_prx + "_by_word"), 'r',
                         encoding='utf-8').readlines()
    data_lines = tqdm(lines,desc="building")
    for line in data_lines:
        chat_corpus.fit(line.strip().split())

    chat_corpus.build_vocab_chat(min_count,max_count,max_feature)
    if(is_target):

        pickle.dump(chat_corpus,open(config.chatboot_config.get("word_corpus"+middle_prx+"_by_word_target"),'wb'))
    else:

        pickle.dump(chat_corpus, open(config.chatboot_config.get("word_corpus" + middle_prx + "_by_word_input"), 'wb'))

if __name__ == '__main__':
    chat_corpus = Chat_corpus()
    compute_build(chat_corpus,fixed=True,min_count=5,by_word=False,is_target=True)













