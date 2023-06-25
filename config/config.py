"""
just configuration for this project to run
"""

import pickle


auto_fix = True

jieba_config = {
    "word_dict":"./../../data/word/word40W.txt",
    "stop_dict":"./../../data/word/stopWordBaiDu.txt",
}

data_path = {
    "xiaohuangji": "./../../data/XiaoHuangJi50W.conv",
    "QA": "./../../data/QA5W.json"
}

process_save_classfiy = {
    "xiaohuangji":"./../../data/classfiy/xiaohuangji.txt",
    "QA":"./../../data/classfiy/QA.txt",
    "classfiy":"./../../data/classfiy/classfiy.txt",
    "classfiy_model":"./../../data/classfiy/classfiy_model.model"
}



"""
Encoder and Decoder using same config params in here
"""
chatboot_config = {

    "target_path_no_by_word":"./../../data/chat/target_no_by_word.txt",
    "input_path_no_by_word": "./../../data/chat/input_no_by_word.txt",
    "word_corpus_no_by_word_input":"./../../data/chat/word_corpus_input_no_by_word.pkl",
    "word_corpus_no_by_word_target":"./../../data/chat/word_corpus_target_no_by_word.pkl",

    "target_path_by_word": "./../../data/chat/target_by_word.txt",
    "input_path_by_word": "./../../data/chat/input_by_word.txt",
    "word_corpus_by_word_input": "./../../data/chat/word_corpus_input_by_word.pkl",
    "word_corpus_by_word_target": "./../../data/chat/word_corpus_target_by_word.pkl",

    "seq2seq_model_no_by_word":"./../../data/chat/seq2seq_model_no_by_word.pth",
    "optimizer_model_no_by_word":"./../../data/chat/optimizer_model_no_by_word.pth",

    "seq2seq_model_by_word": "./../../data/chat/seq2seq_model_by_word.pth",
    "optimizer_model_by_word": "./../../data/chat/optimizer_model_by_word.pth",

    "batch_size": 128,
    "collate_fn_is_by_word": False,

    "input_max_len":12,
    "target_max_len": 12,
    "out_seq_len": 15,
    "dropout": 0.3,
    "embedding_dim": 300,
    "padding_idx": 0,
    "sos_idx": 2,
    "eos_idx": 3,
    "unk_idx": 1,
    "num_layers": 2,
    "hidden_size": 128,
    "bidirectional":True,
    "batch_first":True,
    # support 0,1,..3(gpu) and cup
    "drive":"0",
    "num_workers":0,
    "teacher_forcing_ratio": 0.1,
    # just support "dot","general","concat"
    "attention_method":"general",
    "use_attention": True,
    "beam_width": 3,
    "max_norm": 1,
    "beam_search": True

}


def chat_load_(path,by_word,is_target,fixed=True, min_count=5):
    from corpus.chatbot_corpus.build_chat_corpus import Chat_corpus, compute_build
    after_fix = False
    ws = None
    try:
        ws = pickle.load(open(path, 'rb'))
    except:
        if (auto_fix):
            print("fixing...")
            chat_corpus = Chat_corpus()
            compute_build(chat_corpus=chat_corpus, fixed=fixed, min_count=min_count,
                          by_word=by_word, is_target=is_target)
            after_fix = True

    if (after_fix):
        ws = pickle.load(open(path, 'rb'))
    return ws

def word_corpus_no_by_word_input_load():
    path = chatboot_config.get("word_corpus_no_by_word_input")
    return chat_load_(path,is_target=False,by_word=False)

def word_corpus_no_by_word_target_load():
    path = chatboot_config.get("word_corpus_no_by_word_target")
    return chat_load_(path,is_target=True,by_word=False)

def word_corpus_by_word_input_load():
    path = chatboot_config.get("word_corpus_by_word_input")
    return chat_load_(path,is_target=False,by_word=True)

def word_corpus_by_word_target_load():
    path = chatboot_config.get("word_corpus_by_word_target")
    return chat_load_(path,is_target=True,by_word=True)

chatboot_config_load = {
    "word_corpus_no_by_word_input_load": word_corpus_no_by_word_input_load(),
    "word_corpus_no_by_word_target_load": word_corpus_no_by_word_target_load(),
    "word_corpus_by_word_input_load": word_corpus_by_word_input_load(),
    "word_corpus_by_word_target_load": word_corpus_by_word_target_load(),
}

