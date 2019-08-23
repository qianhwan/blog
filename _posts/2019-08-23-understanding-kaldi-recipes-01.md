---
layout: post
title:  "Understanding kaldi recipes with mini-librispeech example"
date:   2019-08-23 07:15:00 -0400
categories: speech-recognition
tags: kaldi asr
---

# Understanding kaldi recipes with mini-librispeech example
This note provides a high-level understanding of how kaldi recipe scripts work, with the hope that people with little experience in shell scripts (like me) can save some time learning kaldi.

Mini-librispeech is a small subset of LibriSpeech corpus which consists of audio book reading speech. We will go through each step in *kaldi/egs/mini_librispeech/s5/run.sh*.


## Parameters and environment setup
```sh
# Change this location to somewhere where you want to put the data.
data=./corpus/

data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=0
. utils/parse_options.sh

set -euo pipefail

mkdir -p $data
```
`data=./corpus/` specifies where you want to store audio and language model data.

`data_url=www.openslr.org/resources/31` specifies the url for downloading audio data.

`lm_url=www.openslr.org/resources/11` specifies the url for downloading vocabulary, lexicon and pre-trained language model (trained on LibriSpeech).

`. ./cmd.sh` runs script `cmd.sh`, you need to change `queue.pl` to `run.pl` if `GridEngine` is not installed.

`. ./path.sh` runs script `path.sh` which adds all kaldi executable dependencies to your environment path. This is required every time you start a new terminal, and it can avoided by adding all paths in your `.bashrc`.

`stage=0` sets which stage this script is on, you can set it to the stage number that has already been executed to avoid running the same command repeatedly.

`. utils/parse_options.sh` enables argument parsing to kaldi scripts (e.g. `./run.sh --stage 2` sets variable `stage` to 2).

`set -eup pipefail` makes the scripts exit safely when encountering an error.

`mkdir -p $data` creates the data folder (`./corpus/` in this case) if it doesn't exist already.


## Stages
Each kaldi recipe consists of multiple **stages**, which can be spotted with the following syntax:
```sh
if [ $stage -le x ]; then
  ...
fi
```
which simply means run the commands in this block if `stage` is less than or equal to number x. I personally like to change `-le` to `eq` (which means equal) so that I can run the recipe step by step.

`stage` is set to 0 by default, which means the recipe will run all blocks. If you encounter an error, you can check which stages are successfully passes and re-run the recipe by `./run.sh --stage x`.


## Stage 0: data fetching
```sh
for part in dev-clean-2 train-clean-5; do
  local/download_and_untar.sh $data $data_url $part
done
```
Download `dev-clean-2` (dev set) and `train-clean-5` (train set) from the url specified before to `./corpus/` and unzip them. You can check the files in `./corpus/` folder after running.
[screenshot here]

```sh
local/download_lm.sh $lm_url $data data/local/lm
```
This line downloads the pre-trained language model to `./corpus/` then makes a soft link to `data/local/lm`.
[screenshot here]

The files that are downloaded are:
- *3-gram.arpa.gz*, trigram arpa LM.
- *3-gram.pruned.1e-7.arpa.gz*, pruned (with threshold 1e-7) trigram arpa LM.
- *3-gram.pruned.3e-7.arpa.gz*, pruned (with threshold 3e-7) trigram arpa LM.
- *librispeech-vocab.txt*, 200K word vocabulary for the LM.
- *librispeech-lexicon.txt*, pronunciations, some of which G2P auto-generated, for all words in the vocabulary.


## Stage 1: data preparing and LM training
```sh
for part in dev-clean-2 train-clean-5; do
  # use underscore-separated names in data directories.
  local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
done
```
Create all files that are needed for kaldi training (see [here](http://kaldi-asr.org/doc/data_prep.html#data_prep_data) for more details on data preparation). Normally each kaldi recipe comes with a different data preparation script, they creates same files for different dataset. If you want to train a model with your own dataset, you will need to write your own data preparation script that gives you the right *kaldi-style* data files.

If you check `data/train_clean_5` after finishing the above commands, you will see the following text files:
- *wav.scp*, maps wav files to their paths (with some audio processing commands sometime).
- *utt2spk*, maps utterances to their speaker, when speaker information is unknown, we treat each utterance as a new speaker.
- *spk2utt*, maps speakers to the utterances spoken by them.
- *text*, maps recordings to their transcribed text.
- *spk2gender*, maps speakers to their genders.
- *utt2dur*, maps utterances to their durations.
- *utt2num_frames*, maps utterances to their number of frames.

Each data set (train, dev, test) should have their own set of files. Among these files, *wav.scp*, *utt2spk*, *spk2utt* and *text* are essential for building any kaldi models.


```sh
local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
  data/local/lm data/local/lm data/local/dict_nosp
```
*'nosp' refers to the dictionary before silence probabilities and pronunciation*.

Generate silence phones, non-silence phones and optional silence phones. Generated files are as follows:
- *extra_questions.txt*, list of extra questions which will be included in addition to the automatically generated questions for [decision trees](https://kaldi-asr.org/doc/tree_externals.html).
- *lexicon.txt*, sorted lexicon with some additional silence phones.
- *lexiconp.txt*, lexicon with pronunciation probabilities.
- *lexicon_raw_nosil.txt*, the same lexicon.
- *nonsilence_phones.txt*, list of non-silence phones.
- *optional_silence.txt*, list of optional silence phones.
- *silence_phones.txt*, list of silence phones.
More detailed explanation can be found [here](https://kaldi-asr.org/doc/data_prep.html#data_prep_lang_creating)

```sh
utils/prepare_lang.sh data/local/dict_nosp \
  "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
```
This prepares the *lang* directory with the following files:
- *L.fst*, FST form of lexicon.
- *L_disambig.fst*, L.fst but including the [disambiguation symbols](https://kaldi-asr.org/doc/graph.html#graph_disambig).
- *oov.int*, mapped integer of out-of-vocabulary words.
- *oov.txt*, out-of-vocabulary words.
- *phones.txt*, maps phones with integers.
- *topo*, the topology of the HMMs we use.
- *words.txt*, maps words with integers.
- *phones/*, specifies varies things about the phone set.

```sh
local/format_lms.sh --src-dir data/lang_nosp data/local/lm
```
Use *data/lang_nosp/word.txt* format two pruned arpa LMs to *G.fst* in *data/lang_nosp_test_tgmed* and *data/lang_nosp_test_tgsmall*.

```sh
utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
  data/lang_nosp data/lang_nosp_test_tglarge
```
Create ConstArpaLm format language model ( *G.carpa* ) from the full 3-gram arpa LM.


## Stage 2: MFCC extraction
`mfccdir=mfcc` specifies where to store the extracted MFCCs

```sh
for part in dev_clean_2 train_clean_5; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
  steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done
```
Extract MFCCs and computes CMVN stats from *data/dev_clean_2* and *data/train_clean_5* to *mfcc* using 10 parallel jobs. Logs can be found in *exp/make_mfcc*, they are what you are going to check if something goes wrong.

```sh
# Get the shortest 500 utterances first because those are more likely
# to have accurate alignments.
utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short
```
Create a data subset of the shortest 500 utterances. We are not copying any MFCC here, if you look into *data/train_500short* you can find a *feat.scp* that maps the utterances to where their MFCCs are stored.


## Stage 3: monophone training
```sh
steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
  data/train_500short data/lang_nosp exp/mono
```
Train a monophone system using the shortest 500 utterances and the LM trained before, the trained model and logs can be found in *exp/mono*.
`--boost-silence 1.25` sets the factor by which to boost silence likelihoods in alignment to 1.25.
`-nj 5` sets the number of parallel jobs to 5.

```sh
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/mono exp/mono/graph_nosp_tgsmall
  for test in dev_clean_2; do
    steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
      data/$test exp/mono/decode_nosp_tgsmall_$test
  done
)&
```
Create the final graph ( HCLG.fst model ) and decodes *data/dev_clean_2* using the graph. You can find WERs in *exp/mono/decode_nosp_tgsmall_dev_clean_2*.

 In mini_librispeech recipe each training stage (monophone, triphone, dnn etc.) comes with a decoding step, you can comment them out if you don't want to decode with certain models since it takes some time. But it is a good practice to see improvements when the model gets more complicated.

As you can see in *exp/mono/decode_nosp_tgsmall_dev_clean_2*, there are more than one WER file (e.g. *wer_10_0.5*). This is because *steps/decode.sh* calls *local/score.sh* where we play with some scoring parameters to get the best WER.

In the example of *wer_10_0.5*, 10 is the LM-weight for lattice rescoring, 0.5 is the word insertion penalty factor.

```sh
steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
  data/train_clean_5 data/lang_nosp exp/mono exp/mono_ali_train_clean_5
```
Compute the training alignments using the monophone model.


## Stage 4: delta + delta-delta triphone training
```sh
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
  2000 10000 data/train_clean_5 data/lang_nosp exp/mono_ali_train_clean_5 exp/tri1
```
Train a triphone model with MFCC + delta + delta-delta features, using the training alignments generated in **Stage 3**.

*I skipped the decoding commands here.*

```sh
steps/align_si.sh --nj 5 --cmd "$train_cmd" \
  data/train_clean_5 data/lang_nosp exp/tri1 exp/tri1_ali_train_clean_5
```
Compute the training alignments using the triphone model.


## Stage 5: LDA + MLLT triphone training
```sh
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" 2500 15000 \<Paste>
  data/train_clean_5 data/lang_nosp exp/tri1_ali_train_clean_5 exp/tri2b
```
Train a triphone model with LDA and MLLT feature transforms, using the training alignments generated in **Stage 4**.

```sh
steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
  data/train_clean_5 data/lang_nosp exp/tri2b exp/tri2b_ali_train_clean_5
```
Again, compute the training alignments using the newly trained triphone model.


## Stage 6: LDA + MLLT + SAT triphone training
```sh
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
  data/train_clean_5 data/lang_nosp exp/tri2b_ali_train_clean_5 exp/tri3b
```
Train a triphone model with Speaker Adaptation Training, using the training alignments generated in **Stage 5**.

## Stage 7: re-create language model and compute the alignments from SAT model
```sh
steps/get_prons.sh --cmd "$train_cmd" \
  data/train_clean_5 data/lang_nosp exp/tri3b
```
There are several things happen in this command:
- Linear lattices (single path) are generated for each utterance in *train_clean_5* using the latest alignment and LM.
- A bunch of *pron.x.gz* is created with the format of

    `<utterance-id> <begin-frame> <num-frames> <word> <phone1> <phone2> ... <phoneN>`
- Get *pron_counts_nowb.txt* which contains the counts of pronunciations (generated by aligning training data, not from the original text).

```sh
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict_nosp \
  exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
  exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict
```
Take the pronunciation counts and create a modified dictionary directory with pronunciation probabilities.

```sh
utils/prepare_lang.sh data/local/dict \
  "<UNK>" data/local/lang_tmp data/lang

local/format_lms.sh --src-dir data/lang data/local/lm

utils/build_const_arpa_lm.sh \
  data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
```
Then we build a new ConstArpa LM with the new dictionary.

```sh
steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
  data/train_clean_5 data/lang exp/tri3b exp/tri3b_ali_train_clean_5
```
Compute the training alignments using the SAT model and new *L.fst*.


## Stage 8: generating graphs and decoding
```sh
utils/mkgraph.sh data/lang_test_tgsmall \
  exp/tri3b exp/tri3b/graph_tgsmall
```
Create the final graph (HCLG.fst model) with the small trigram LM.

```sh
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3b/graph_tgsmall data/$test \
  exp/tri3b/decode_tgsmall_$test
```
Decode *test* set using the SAT model and the small trigram LM, WERs can be found at *exp/tri3b/decode_tgsmall_dev_clean_2*.

```sh
steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
  data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test
```
Re-score decoded lattice ( *exp/tri3b/decode_tgsmall_dev_clean_2* ) with medium trigram LM, lattices and WERs after re-scoring can be found at *exp/tri3b/decode_tgmed_dev_clean_2*.

```sh
steps/lmrescore_const_arpa.sh \
  --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
  data/$test exp/tri3b/decode_{tgsmall,tglarge}_$test
```
Re-score decoded lattice ( *exp/tri3b/decode_tgmed_dev_clean_2* ) with large ConstArpa LM, lattices and WERs after re-scoring can be found at *exp/tri3b/decode_tglarge_dev_clean_2*.

You can see the WER improvements from *exp/mono/decode_nosp_tgsmall_dev_clean_2* to *exp/tri3b/decode_tglarge_dev_clean_2*

## Stage 9: DNN training
I'll leave this to another note.
