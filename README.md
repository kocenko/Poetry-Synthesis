# Poetry-Synthesis
Generatively Pretrained Transformer generating poetry. 

## Introduction
The idea for this project came from [the video](https://www.youtube.com/watch?v=kCc8FmEb1nY) of Andrej Karpathy explaining, how Generatively Pretrained Transformers are created. In the lecture presented by Andrej, the [tinyshakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) was used. As an additional exercise, he suggested using different dataset. This project is the response to this suggestion. Moreover, different tokenization algorithms might be employed to explore its effect on the outcome.

## Dataset
Inspired by the idea of using poems or stories of someone whose incredible work is in the public domain, we used [the dataset](https://www.kaggle.com/datasets/leangab/poe-short-stories-corpuscsv) containing short stories of the great American novelist, Edgar Allan Poe. As the input needs to be a single text file, only text of the novels was extracted from the file and placed into the text file.