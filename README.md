# GPT - Character-based Text Generation

This project focuses on **generating text** character-by-character using **Transformer neural networks**. The implementation involves training a model on a dataset, specifically on Shakespeare's works, to produce coherent and stylistically consistent text. The project demonstrates the application of deep learning techniques in **natural language processing (NLP)** for creative **text generation**.

![Deep Learning](https://img.shields.io/badge/Skill-Deep%20Learning-yellow)
![PyTorch](https://img.shields.io/badge/Skill-PyTorch-blueviolet)
![Transformers](https://img.shields.io/badge/Skill-Deep%20Learning-orange)
![Generative AI](https://img.shields.io/badge/Skill-Generative%20AI-green)
![Model Training and Evaluation](https://img.shields.io/badge/Skill-Model%20Training%20and%20Evaluation-orange)
![Model Deployment](https://img.shields.io/badge/Skill-Model%20Deployment-purpule)
![CI/CD](https://img.shields.io/badge/Skill-CI/CD-blue)
![Web Application](https://img.shields.io/badge/Skill-Web%20Application-yellow)
![Python Programming](https://img.shields.io/badge/Skill-Python%20Programming-blue)

## install
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- pytorch 
- numpy 
- tiktoken for OpenAI's fast BPE code 
- wandb for optional logging 
- tqdm for progress bars 

## How to Run
First, we train a character-level GPT on the works of Shakespeare and download it as a single (1MB) file and turn it from raw text into one large stream of integers
```bash
python data/shakespeare_char/prepare.py
```
This creates a train.bin and val.bin in that data directory. Now it is time to train your GPT. Then, we can quickly train a GPT with the settings provided in the config/train_shakespeare_char.py config file using GPU.

```bash
python train.py config/train_shakespeare_char.py
```
If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. 
So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```bash
python sample.py --out_dir=out-shakespeare-char
```
This generates a few samples, for example:

```bash
How cheerful rather and her eyes
Which for that I am here in the people,
Having been commended and the roses of such births,
Shows forth, if thou the sickly be enough,
And not have no more wrong'd with his own.
Nor who is yours of royal pity of Buckingham,
Which you for his land-fortune with us,
So virtuous like prayers for our concils
Full of his foes, that all dearame mercy,
Your noble was a wife of your graces
With either dearth, your hardy devise,
And the itself of her brother, I did not see
---------------

She was served his blood against his sheep-sequing winds,
A man to her good man in her to speak.

DUKE VINCENTIO:
Nor here, not to see him of death:
You rage it, good my order is not my lord.

DUKE VINCENTIO:
Say you to your purse nor of your grace and have been
To prison, my sister, but who is the may be with joy.

LUCENTIO:
Have you here all father too.

DUKE VINCENTIO:
Be no pretties a barren?

DUKE VINCENTIO:
Marry you, my lord; but you will pard your gates of suit.

ANGELO:
And little for y
---------------

I have hence, lord, tell him her mind;
And there a devision cannot be gone,
To save the man you and bow of my love,
As you will written yourself, will I wish
You have done: but yet I have done so,
So pray your voices in my hands, whereof,
Know her to your eyes and sharp tears.

JULIET:
And rather have you quite him dead!

Nurse:
Here's gone, whom your ransoment; and so we are more
To make the halber-day to make a child-blood,--O brave me!--
And as it is your son, sir, when you must be discover'd

```


