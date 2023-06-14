SOS_Token = 0
EOS_Token = 1

class Lang:
    
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.nwords = 2 # It includes both EOS and SOS

    
    def addSentence(self, sentence):
        for word in sentence.spit(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.nwords
            self.word2count[word] = 1
            self.index2word[self.nwords] = word
            self.nwords += 1
        else:
            self.nwords += 1