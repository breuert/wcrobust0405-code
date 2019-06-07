# IRC-CENTRE2019 Docker Image

TODO: Travis

[**Timo Breuer**](https://github.com/breuert/) and [**Philipp Schaer**](https://github.com/phschaer/)

This is the docker image for our replicated submission to [CENTRE@CLEF2019](http://www.centre-eval.org/clef2019/) conforming to the [OSIRRC jig](https://github.com/osirrc/jig/) for the [Open-Source IR Replicability Challenge (OSIRRC 2019) at SIGIR 2019](https://osirrc.github.io/osirrc2019/).
This image is available on [Docker Hub](https://hub.docker.com/r/osirrc2019/irc-centre2019) has been tested with the jig at commit [ca31987](https://github.com/osirrc/jig/commit/ca3198704795f2b6de8b78ed7a66bbdf1dccadb1) (6/5/2019).

+ Supported test collections: `core17`
+ Required training collections: `robust04`, `robust05`
+ Supported hooks: `init`, `index`, `search`

## Quick Start

Use the commands below to get the runs for WCRobust04 and WCRobust0405 as they were
replicated in the course of our participation in [CENTRE@CLEF19](http://www.centre-eval.org/clef2019/).
The following `jig` command can be used to index the New York Times corpus and prepare training data for WCRobust04:

```
python run.py prepare \
    --repo osirrc2019/irc-centre2019 \
    --collections robust04=/path/to/robust04/=trectext \
                  core17=/path/to/core17/=trectext \
    --opts run="wcrobust04"
```

The argument `run` can be customized to `run`="wcrobust0405" in order to prepare training data for WCRobust0405.
In this case, the `robust05` corpus has to be mounted as an additional volume.

```
python run.py prepare \
    --repo osirrc2019/irc-centre2019 \
    --collections robust04=/path/to/robust04/=trectext \
                  robust05=/path/to/robust05/=trectext \
                  core17=/path/to/core17/=trectext \
    --opts run="wcrobust0405"
```

The following `jig` command can be used to perform a retrieval run on the New York Times depending on the previously defined training corpora.

```
python run.py search \
    --repo osirrc2019/irc-centre2019 \
    --collection core17 \
    --topic topics/topics.core17.txt \
    --output /path/to/output/ \
    --qrels qrels/qrels.core17.txt
```

## Expected Results

TODO: add outcomes

## Implementation

### Dockerfile

### init

### index

### search

