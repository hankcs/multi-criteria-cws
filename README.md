# multi-criteria-cws
Codes and corpora for paper "[Effective Neural Solution for Multi-Criteria Word Segmentation](https://arxiv.org/abs/1712.02856)" (accepted & forthcoming at SCI-2018).

### Dependency

* Python3
* [DyNet==2.0.1](https://github.com/hankcs/multi-criteria-cws/issues/1)

## Quick Start

Run following command to prepare corpora, split them into train/dev/test sets etc.:

```bash
python3 convert_corpus.py 
```

Then convert a corpus `$dataset` into pickle file:

```bash
./script/make.sh $dataset
```

* `$dataset` can be one of the following corpora: `pku`, `msr`, `as`, `cityu`, `sxu`, `ctb`, `zx`, `cnc`, `udc` and `wtb`.
* `$dataset` can also be a joint corpus like `joint-sighan2005` or `joint-10in1`.
* If you have access to sighan2008 corpora, you can also make `joint-sighan2008` as your `$dataset`.

Finally, one command performs both training and test on the fly:

```bash
./script/train.sh $dataset
```

## Performance

### sighan2005

![sighan2005](http://wx4.sinaimg.cn/large/006Fmjmcly1fm8ru5refwj31960ssah6.jpg)

### sighan2008
  
![sighan2008](http://wx4.sinaimg.cn/large/006Fmjmcly1fm8rakv137j31in0petfw.jpg)

### 10-in-1

Since SIGHAN bakeoff 2008 datasets are proprietary and difficult to obtain, we decide to conduct additional experiments on more freely available datasets, for the public to test and verify the efficiency of our method. We applied our solution on 6 additional freely available datasets together with the 4 sighan2005 datasets.

![10in1](http://wx1.sinaimg.cn/large/006Fmjmcly1fm5vnkn5zxj31h00ik0z2.jpg)


## Corpora

In this section, we will briefly introduce those corpora used in this paper.

### 10 corpora in this repo

Those 10 corpora are either from official sighan2005 website, or collected from open-source project, or from researchers' homepage. Licenses are listed in following table.

![licence](http://wx3.sinaimg.cn/large/006Fmjmcly1fm6jtha3tmj318r0l40x9.jpg)


### sighan2008

As sighan2008 corpora are proprietary, we are unable to distribute them. If you have a legal copy, you can replicate our scores following these instructions.

Firstly, link the sighan2008 to data folder in this project.

```
ln -s /path/to/your/sighan2008/data data/sighan2008
```

Then, use [HanLP](https://github.com/hankcs/HanLP) for Traditional Chinese to Simplified Chinese conversion, as shown in the following Java code snippets:

```java
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(
            "data/sighan2008/ckip_seg_truth&resource/ckip_truth_utf16.seg"
        ), "UTF-16"));
        String line;
        BufferedWriter bw = IOUtil.newBufferedWriter(
            "data/sighan2008/ckip_seg_truth&resource/ckip_truth_utf8.seg");
        while ((line = br.readLine()) != null)
        {
            for (String word : line.split("\\s"))
            {
                if (word.length() == 0) continue;
                bw.write(HanLP.convertToSimplifiedChinese(word));
                bw.write(" ");
            }
            bw.newLine();
        }
        br.close();
        bw.close();
```
You need to repeat this for the following `4` files:

1. ckip_train_utf16.seg
2. ckip_truth_utf16.seg
3. cityu_train_utf16.seg
4. cityu_truth_utf16.seg

Then, uncomment following codes in `convert_corpus.py`:

```python
    # For researchers who have access to sighan2008 corpus, use official corpora please.
    print('Converting sighan2008 Simplified Chinese corpus')
    datasets = 'ctb', 'ckip', 'cityu', 'ncc', 'sxu'
    convert_all_sighan2008(datasets)
    print('Combining those 8 sighan corpora to one joint corpus')
    datasets = 'pku', 'msr', 'as', 'ctb', 'ckip', 'cityu', 'ncc', 'sxu'
    make_joint_corpus(datasets, 'joint-sighan2008')
    make_bmes('joint-sighan2008')
```

Finally, you are ready to go:

```
python3 convert_corpus.py
./script/make.sh joint-sighan2008
./script/train.sh joint-sighan2008
```

## Acknowledgments

- Thanks for those friends who helped us with the experiments.
- Credits should also be given to those generous researchers who shared their corpora with the public, as listed in license table. Your datasets indeed helped those small groups (like us) without any funding.
- Model implementation modified from a Dynet-1.x version by [rguthrie3](https://github.com/rguthrie3/BiLSTM-CRF).



